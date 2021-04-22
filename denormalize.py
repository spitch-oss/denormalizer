#!/usr/bin/env python3

"""
Denormalize strings or files with pretrained models.
"""

import argparse
import sys

from fairseq.models.transformer import TransformerModel
from fairseq.models.transformer import base_architecture
import scripts.custom_arch


def denormalize_file(DENORMALIZER, file, beam, outfile):
    """
    Denormalize all lines in a file.

    Args:
        DENORMALIZER: Fairseq model.
        file (str): File with normalized sequences.
        beam (int): Beam size for decoding.
        outfile (str): Output file for the denormalized sequences.

    Returns:
        None.
    """

    n_sents = 0

    out = (sys.stdout if outfile == sys.stdout else open(outfile, 'w'))

    with open(file) as normalized:
        for line in normalized:
            denormalized = DENORMALIZER.translate(line.strip(), beam=beam)
            n_sents += 1
            print(denormalized, file=out)

    print(f'\nProcessed {n_sents} sentences.', file=sys.stdout)
    out.close()


def denormalize_string(DENORMALIZER, string, beam, outfile):
    """
    Denormalize a string.

    Args:
        DENORMALIZER: Fairseq model.
        string (str): A normalized sequence.
        beam (int): Beam size for decoding.
        outfile (str): Output file for the denormalized sequences.

    Returns:
        None.
    """

    denormalized = DENORMALIZER.translate(string, beam=beam)

    out = (sys.stdout if outfile == sys.stdout else open(outfile, 'w'))
    print(denormalized, file=out)
    out.close()


def denormalize_interactively(DENORMALIZER, beam):
    """
    Denormalize all lines in a file.

    Args:
        DENORMALIZER: Fairseq model.
        beam (int): Beam size for decoding.

    Returns:
        None.
    """

    print('Starting interactive mode.\nEnter a normalized string:\n')
    while True:
        string = input('I: ')
        denormalized = DENORMALIZER.translate(string, beam=beam)
        print('O:', denormalized, '\n')


def add_args(parser):
    """
    Add arguments to the CL parser.

    Args:
        parser: argparse.ArgumentParser() object.

    Returns:
        None
    """

    parser.add_argument('--lang', '-l',
                        help='language of the model',
                        default='en')
    parser.add_argument('--size',
                        help='size of the model',
                        default='l')
    parser.add_argument('--beam', '-b', type=int,
                        help='beam size',
                        default=5)
    parser.add_argument('--detokenize', action='store_true',
                        help='detokenize output')
    parser.add_argument('--outfile', '-o', default=sys.stdout,
                        help='file to which the denormalized strings '\
                        'will be written (default: sys.stdout)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--string', '-s', help='string to denormalize')
    mode.add_argument('--file', '-f', help='file to denormalize')


def main():
    """
    Parse CL arguments and and denormalize the input.
    """

    description = 'Denormalize normalized strings.'
    parser = argparse.ArgumentParser(description=description)
    add_args(parser)

    args = parser.parse_args()

    DENORMALIZER = TransformerModel.from_pretrained(
        'models',
        checkpoint_file=f'model_{args.lang}_{args.size}.pt',
        data_name_or_path=f'data-bin/{args.lang}',
        bpe='subword_nmt',
        bpe_codes=f'data-bin/{args.lang}/bpe_code',
        tokenizer=('moses' if args.detokenize else None))

    if args.string:
        denormalize_string(DENORMALIZER, args.string, args.beam, args.outfile)
    elif args.file:
        denormalize_file(DENORMALIZER, args.file, args.beam, args.outfile)
    else:
        denormalize_interactively(DENORMALIZER, args.beam)


if __name__ == '__main__':
    main()
