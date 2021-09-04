#!/usr/bin/env python3
# coding: utf8


"""
Compute error rates, given a reference corpus and a hypothesis corpus
and print evaluation report to stdout or an output file.
"""

import argparse
import json
import os
import re
import sys
import time
from itertools import islice

from fairseq.models.transformer import TransformerModel
from fairseq.models.transformer import base_architecture

import edit_distance

os.sys.path.append(os.path.abspath("../.."))
from denormalizer.scripts import custom_arch


def generate_hypotheses(args):
    """
    Generate hypotheses for all input sequences and write them to a file.

    Args:
        args: all args passed through the command line.

    Note:
        The generated hypotheses are written to the output file specified
        in args and not returned.

    Returns:
        tuple: (total_dur, n_sents)

    """

    total_dur = 0
    n_sents = 0

    DENORMALIZER = TransformerModel.from_pretrained(
        args.checkpoint_dir,
        checkpoint_file=args.checkpoint_file,
        data_name_or_path=args.data_name_or_path,
        bpe='subword_nmt', bpe_codes=args.bpe_codes,
        beam=args.beam)

    if args.use_gpu:
        DENORMALIZER.cuda()

    batch_size = 64

    with open(args.source) as source, open(args.hypothesis, 'w') as hyp:
        for lines in iter(lambda: tuple(islice(source, batch_size)), ()):
            lines = [l.strip() for l in lines]
            start = time.time()
            punct = DENORMALIZER.translate(lines, beam=args.beam)
            end = time.time()
            dur = end - start
            total_dur += dur
            n_sents += batch_size
            for line in punct:
                hyp.write(line+'\n')
            print(f'Processed {n_sents} sentences.\r', end='')

    print(f'Processed {n_sents} sentences.\r', end='')
    print('\n')
    return total_dur, n_sents


def print_report(results, duration, n_sents, config, outfile=sys.stdout):
    """
    Print report with various pre-calculated error rates.

    Args:
        results (tuple): A tuple with a sub-tuple for each error type.
            Each sub-tuple should contain the name of the error type,
            the error count, the total number of words, and the error
            rate.
        duration (float): Total generation time.
        n_sents (int): Number of sentences.
        config (dict): A dictionary with character and symbol sets.
        outfile: an open file where the report is written to.
            Defaults to sys.stdout.

    Returns:
        None.
    """

    table = []
    table.append('-'*43)
    table.append('{0:<16}{1:>8}{2:>10}{3:>9}'
                 .format('TYPE', 'ERRORS', 'TOTAL', 'RATE'))
    table.append('-'*43)
    for i, line in enumerate(results):
        if i == 2:
            table.append('-'*43)
        table.append('{0:<16}{1:>8}{2:>10}{3:>8.2f}%'.format(line[0], *line[1]))
    table.append('-'*43)

    table.append('\n'+'-'*43)
    if duration is not None:
        table.append('GENERATION TIME:')
        table.append('-'*43)
        table.append('{0:<16}{1:>27}'.format('Total:', int(duration)))
        table.append('{0:<16}{1:>27}'.format('Avg:', round(duration/n_sents, 2)))
        table.append('-'*43)
        table.append('\n'+'-'*43)
    table.append('CHARACTER SETS:')
    table.append('-'*43)
    table.append('{0:<16}{1:>8}'.format('Copy chars:',
                                        ''.join(config['copy_chars'])))
    table.append('{0:<16}{1:>8}'.format('Punctuation:',
                                        ''.join(config['punctuation'])))
    table.append('{0:<16}{1:>8}'.format('Symbols:',
                                        ''.join(config['symbols'])))
    table.append('-'*43)

    for line in table:
        print(line, file=outfile)


def print_errors(alignments, outfile=sys.stdout):
    """
    Print aligned sequences with errors.

    Args:
        alignments (list): alignments in the format returned by
            edit_distance.get_alignments().
        outfile: an open file where the report is written to.
            Defaults to sys.stdout.

    Returns:
        None.
    """

    print('\nAligned errors:')
    for alignment in alignments:
        for line in alignment:
            print(*line, '\n', file=outfile)


def add_args(parser):
    """
    Add arguments to the CL parser.

    Args:
        parser: argparse.ArgumentParser() object.

    Returns:
        None
    """

    parser.add_argument('--reference', '-r',
                        help='a text file with references',
                        required=True)
    parser.add_argument('--hypothesis', '-y',
                        help='an empty file for the hypotheses',
                        required=True)
    parser.add_argument('--source', '-s',
                        help='a text file with source sentences',
                        required=True)
    parser.add_argument('--lang', '-l',
                        help='a language code to select the appropriate '\
                        'section in the configuration file',
                        required=True)
    parser.add_argument('--config', '-c',
                        help='JSON config file specifying chars, punctuation '\
                             'and symbols for evaluation'\
                             '(default: config.json)',
                        default='config.json')
    parser.add_argument('--outfile', '-o', default=sys.stdout,
                        help='file to which the evaluation report will '\
                        'be written (default: sys.stdout)')
    parser.add_argument('--print-errors', action='store_true',
                        help='print aligned errors')
    parser.add_argument('--outfile-errors', default=sys.stdout,
                        help='file to which the aligned errors will '\
                        'be written in case --print-errors is selected'\
                        '(default: sys.stdout)')
    parser.add_argument('--beam', '-b', type=int, default=5, help='beam size '\
                        ' (default: 5)')
    parser.add_argument('--checkpoint-dir', default='../checkpoints/denorm',
                        help='path to checkpoint dir (default: ../checkpoints/denorm)')
    parser.add_argument('--checkpoint-file', default='checkpoint_best.pt',
                        help='name of checkpoint file '\
                        '(default: checkpoint_best.pt)')
    parser.add_argument('--data-name-or-path', default='../../data-bin/denorm',
                        help='path to directory with binarized data '\
                        '(default: ../../data-bin/denorm)')
    parser.add_argument('--bpe-codes', default='../data-bin/denorm/bpe_code',
                        help='name/path of BPE file '\
                        '(default: ../data-bin/denorm/bpe_code)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='use GPU if available (default: False)')
    parser.add_argument('--force-overwrite', '-f', action='store_true',
                        help='overwrite existing files (default: False)')


def main():
    """
    Parse CL arguments and run evalutation on the given reference
    and hypothesis files.
    """

    example = 'Evaluate the performance of a denormalizer.'
    parser = argparse.ArgumentParser(description=example)
    add_args(parser)

    args = parser.parse_args()

    if args.outfile != sys.stdout:
        if os.path.isfile(args.outfile) and args.force_overwrite is False:
            raise FileExistsError(f'File {args.outfile} exists already. '\
                                  'Please choose another file name or '\
                                  'use the option --force-overwrite.')
        outfile = open(args.outfile, 'w')
    else:
        outfile = sys.stdout

    if args.outfile_errors != sys.stdout:
        if os.path.isfile(args.outfile_errors) \
            and args.force_overwrite is False:
            raise FileExistsError(f'File {args.outfile_errors} exists '\
                                  'already. Please choose another file '\
                                  'name or use the option --force-overwrite.')
        outfile_errors = open(args.outfile_errors, 'w')
    else:
        outfile_errors = sys.stdout

    with open(args.config) as config_file:
        config = json.load(config_file)[args.lang]

    if not os.path.isfile(args.hypothesis):
    # note: the hypotheses themselves are written to a file and not returned
        total_dur, n_sents = generate_hypotheses(args)
    else:
        total_dur, n_sents = None, None

    results = edit_distance.calculate_error_rates(
        args.reference, args.hypothesis, config)

    print_report(results, total_dur, n_sents, config, outfile)
    if args.print_errors:
        alignments = edit_distance.get_alignments(args.reference,
                                                  args.hypothesis)

        print_errors(alignments, outfile_errors)

if __name__ == '__main__':
    main()
