#!/usr/bin/env python3
# coding: utf8

import sys
from re import sub

from nltk.tokenize import word_tokenize
from unidecode import unidecode

nltk.download('punkt')

def is_trash(text):
    """
    Decide whether a sentence is of low quality based on simple heuristics.

    Args:
        text (str): The string to be analyzed.

    Returns:
        bool: True if string is trash, False otherwise.
    """

    if not text.endswith(('.', ':', '!', '?')) or len(text) < 6:
        return True
    if text.count('&') > 3 or text.count(',') > 5:
        return True
    if '...' in text or '|' in text:
        return True
    if sum(char.isdigit() for char in text) > len(text)/2:
        return True
    if sum(len(char.encode()) > 1 for char in text) > len(text)/4:
        return True
    return False


def replace_non_ascii_chars(text):
    charset = "A-Za-záéíóúàèìòùâêîôûäëïöüçñßÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÄËÏÖÜÇÑ'"\
              ",.!?:;- +/%°#&§@0-9"
    exceptions_ = set(c for c in charset if len(c.encode()) > 1)
    return ''.join([unidecode(char) if len(char.encode()) > 1
                    and char not in exceptions_ else char for char in text])


def _normalize_date(d, m, y):
    if int(y) > 30:
        y = '19'+y
    else:
        y = '20'+y
    return f'{y}-{m}-{d}'


def normalize_dates(text):
    text = sub(r'(?<![\d\-])(\d\d)-(\d\d)-(\d\d\d\d)(?![\d\-])',
               r'\3-\2-\1', text)
    text = sub(r'(?<![\d\-])(\d\d)-(\d\d)-(\d\d)(?![\d\-])',
               lambda x: _normalize_date(x.group(1), x.group(2), x.group(3)),
               text)
    return text


def correct(text):
    text = sub(r'["\(\)]', '', text)
    text = sub(r"(?<![a-z])'", '', text)
    text = sub(r"'(?![a-z])", '', text)
    text = sub(r'(?<=\d),(?=\d)', '', text)
    text = sub(r'(?<=\d)(?![\d:\-\.])', ' ', text)
    text = sub(r'(?![\d:\-\.])(?=\d)', ' ', text)
    # normalize dashes and slashes
    text = sub(r' ?/ ?', ' / ', text)

    text = normalize_dates(text)

    # remove unnecessary exclamation marks
    text = sub(r'^([\w\d:\-,]+( [\w\d:\-,]+){8,}) ?!', r'\1.', text)
    text = sub(r'(^|\.|\?|!|:| -)(( \S+){9,})!', r'\1\2.', text)
    text = sub(r' ?\- ?', ' - ', text)

    text = sub(r' +', ' ', text).strip()
    text = replace_non_ascii_chars(text)

    # remove punctuation at the beginning of a line and repeated punctuation
    text = sub(r'^[^a-zA-Z§\d]+', '', text)
    text = sub(r"([!\?\.:,;\-'\(\)\[\]] ?)+", r'\1', text)

    months = {
        'Jan': 'January',
        'Feb': 'February',
        'Mar': 'March',
        'Apr': 'April',
        'Jun': 'June',
        'Jul': 'July',
        'Aug': 'August',
        'Sep': 'September',
        'Oct': 'October',
        'Nov': 'November',
        'Dec': 'December'}

    text = sub(rf'\b({"|".join(months.keys())})\.?(?=( |$))',
               lambda x: months[x.group(1)], text)

    # normalize times
    text = sub(r'(?=\b\d:[0-6]\d\b)', '0', text)
    text = sub(r'([ap]) ?\.? ?m\.?', r'\1m', text)
    text = sub(r'(?<=\d):(?=\d)', ' : ', text)

    text = ' '.join(word_tokenize(text))
    return text


def main():
    print('Preprocessing parallel corpus ...')

    min_length = 7
    num_sents = 0
    num_del = 0

    with open(sys.argv[1]) as source_in, open(sys.argv[2]) as target_in, \
         open(sys.argv[3], 'w') as source_out, open(sys.argv[4], 'w') \
         as target_out:
        for src, tgt in zip(source_in, target_in):
            num_sents += 1
            src, tgt = src.strip(), tgt.strip()
            if is_trash(tgt):
                num_del += 1
            else:
                tgt = correct(tgt)
                if len(tgt) < min_length:
                    num_del += 1
                else:
                    source_out.write(src+'\n')
                    target_out.write(tgt+'\n')
            if num_sents % 1000 == 0:
                print(f'Processed {num_sents} sentences.\r', end='')

    print(f'Processed {num_sents} sentences.\r', end='')
    print(f'\nDone. Deleted {num_del} sentences '
          f'({round((num_del/num_sents)*100, 2)}%).')

if __name__ == '__main__':
    main()
