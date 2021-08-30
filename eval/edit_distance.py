#!/usr/bin/env python3
# coding: utf8


"""
This script contains functions to calculate error rates,
including word error rate, sentence error rate, and more specific
error rates such as digit error rate.
"""


def sentence_error(source, target):
    """
    Evaluate whether the target is identical to the source.


    Args:
        source (str): Source string.
        target (str): Target string.

    Returns:
        int: 0 if the target is equal to the source, 1 otherweise.
    """

    return 0 if target == source else 1


def count_word_errors(source, target):
    """
    Calculate the number of edit operations between a target
    and the source.

    Args:
        source (str): Source string.
        target (str): Target string.

    Returns:
        tuple (int, int): A tuple containing the number of word errors
            and the total number of words in the source.
    """

    error_count = 0
    for code in edit_operations(source, target):
        if code in 'dis': # delete, insert, substitute
            error_count += 1
    return error_count, len(source)


def prepare_special_error_type(source, target, error_type, charset=None):
    """
    Remove all non-matching words from both the source and the target string.

    Args:
        source (str): Source string.
        target (str): Target string.
        error_type (str): One of: 'copy_chars', 'uppercase', 'digits',
            'punctuation', 'symbols'. Other strings are ignored and the
            standard word error rate will be calculated.
        charset (set): the set with all characters (or strings)
            to be considered. Defaults to None.

    Returns:
        tuple (list, list): a tuple with the source and the target strings.

    """

    if error_type == 'copy_chars':
        source = [word for word in source if
                  all(c in charset for c in word)]
        target = [word for word in target
                  if all(c in charset for c in word) and word in source]

    elif error_type == 'uppercase':
        source = [word for word in source
                  if any(c.isupper() for c in word)]
        target = [word for word in target
                  if any(c.isupper() for c in word)]

    elif error_type == 'digits':
        source = [word for word in source
                  if any(c.isdigit() for c in word)]
        target = [word for word in target
                  if any(c.isdigit() for c in word)]

    elif error_type in ('punctuation', 'symbols'):
        source = [word for word in source if word in charset]
        target = [word for word in target if word in charset]

    return source, target



def count_special_errors(source, target, error_type, charset=None):
    """
    Calculate the number of edit operations between the target
    and the source for only a subset of all words (e.g. digits).

    Args:
        source (str): Source string.
        target (str): Target string.
        error_type (str): One of: 'copy_chars', 'uppercase', 'digits',
            'punctuation', 'symbols'. Other strings are ignored and the
            standard word error rate will be calculated.
        charset (set): the set with all characters (or strings)
            to be considered. Defaults to None.

    Returns:
        tuple (int, int): A tuple containing the number of word errors
            and the total number of words in the source.
    """

    source, target = prepare_special_error_type(
        source, target, error_type, charset)

    return count_word_errors(source, target)


def edit_operations(source, target):
    """
    Get a list of edit operations for converting the source into the target.

    Args:
        source (str): Source string.
        target (str): Target string.

    Returns:
        list: The list of edit operations.
    """

    matrix = levenshtein_distance(source, target)
    codes = []

    # initiate row and column so that the reading begins
    # at the bottom right corner of the matrix
    row = len(matrix) - 1
    column = len(matrix[0]) - 1

    while row != 0 or column != 0:
        # retrieve value of the current cell and the surrounding cells
        currentcell = matrix[row][column]
        if row != 0 and column != 0:
            diagonal = matrix[row-1][column-1]
            top = matrix[row-1][column]
            left = matrix[row][column-1]
        elif row == 0:
            diagonal = matrix[row][column-1]+1
            top = matrix[row][column]+1
            left = matrix[row][column-1]
        elif column == 0:
            diagonal = matrix[row-1][column]+1
            top = matrix[row-1][column]
            left = matrix[row][column]+1

        # find the lowest value surrounding the current cell
        options = [diagonal, top, left]
        cheapest = options.index(min(options))

        # effect the movement to the cell with the lowest value
        # and add the corresponding code to the list of codes
        if cheapest == 0:
            if diagonal == currentcell:
                codes.insert(0, 'e')
            else:
                codes.insert(0, 's')
            row -= 1
            column -= 1

        elif cheapest == 1:
            codes.insert(0, 'd')
            row -= 1

        elif cheapest == 2:
            codes.insert(0, 'i')
            column -= 1
    return codes


def levenshtein_distance(source, target):
    """
    Get a matrix with local Levenshtein distance optima.

    Args:
        source (str): Source string.
        target (str): Target string.

    Returns:
        list of lists: A matrix with Levenshtein operations.
    """

    n = len(source)
    m = len(target)
    # create matrix with None in every field
    d = [[None for _ in range(m+1)] for _ in range(n+1)]
    # set the top left field to 0
    d[0][0] = 0
    # complete first column and first row
    for i in range(1, n+1):
        d[i][0] = d[i-1][0] + 1
    for j in range(1, m+1):
        d[0][j] = d[0][j-1] + 1

    # fill the rest of the matrix with the minimal value
    # of all possible operations
    # comparing the field to the left, upper left and top
    for i in range(1, n+1):
        for j in range(1, m+1):
            d[i][j] = min(
                d[i-1][j] + 1, # del
                d[i][j-1] + 1, # ins
                d[i-1][j-1] + (1 if source[i-1] != target[j-1] else 0)
                ) # sub
    return d


def error_rate(error_count, total):
    """
    Calculate the error rate, given the error count and the total number
    of words.

    Args:
        error_count (int): Number of errors.
        total (int): Total number of words (of the same type).

    Returns:
        tuple (int, int, float): The error count, the total number
            of words, and the calculated error rate.
    """

    if total == 0:
        return error_count, total, 0.0
    return error_count, total, (error_count/total)*100


def calculate_error_rates(source_file, target_file, config):
    """
    Calculate sentence error rate, word error rate and some additional specific
    word error rates, including punctuation and digit error rates.

    Args:
        source_file (str): File with source strings.
        target_file (str): File with target string.
        config (dict): A dictionary with character and symbol sets.

    Returns:
        tuple of tuples: A tuple containing a tuple for each implemented
        error type. Each sub-tuple contains the name of the error type
        (e.g., 'Punctuation'), the number of errors, the total number of
        tokens, and the error rate.
    """

    with open(source_file) as src_file, open(target_file) as tgt_file:

        sents, sent_errors = 0, 0
        words, word_errors = 0, 0

        copy_words, copy_errors = 0, 0
        upper_words, upper_errors = 0, 0
        digit_words, digit_errors = 0, 0
        punct_words, punct_errors = 0, 0
        symbol_words, symbol_errors = 0, 0

        for src, tgt in zip(src_file, tgt_file):
            src, tgt = src.strip(), tgt.strip()

            sents += 1
            sent_errors += sentence_error(src, tgt)

            src, tgt = src.split(), tgt.split()

            errors, total = count_word_errors(src, tgt)
            word_errors += errors
            words += total

            errors, total = count_special_errors(
                src, tgt, 'copy_chars', set(config['copy_chars']))
            copy_errors += errors
            copy_words += total

            errors, total = count_special_errors(src, tgt, 'uppercase')
            upper_errors += errors
            upper_words += total

            errors, total = count_special_errors(src, tgt, 'digits')
            digit_errors += errors
            digit_words += total

            errors, total = count_special_errors(
                src, tgt, 'punctuation', set(config['punctuation']))
            punct_errors += errors
            punct_words += total

            errors, total = count_special_errors(
                src, tgt, 'symbols', set(config['symbols'].split()))
            symbol_errors += errors
            symbol_words += total

    return (('Sentence', error_rate(sent_errors, sents)),
            ('Word', error_rate(word_errors, words)),
            ('Copy Word', error_rate(copy_errors, copy_words)),
            ('Uppercase Word', error_rate(upper_errors, upper_words)),
            ('Digit', error_rate(digit_errors, digit_words)),
            ('Punctuation', error_rate(punct_errors, punct_words)),
            ('Symbol', error_rate(symbol_errors, symbol_words)))


def get_alignments(source_file, target_file):
    """
    Get word alignments for all pairs of strings which contain a word error.

    Args:
        source_file (str): File with source strings.
        target_file (str): File with target string.

    Returns:
        list: alignment for all pairs of strings which are not identical.
    """

    alignments = []
    with open(source_file) as src_file, open(target_file) as tgt_file:
        for src, tgt in zip(src_file, tgt_file):
            src, tgt = src.strip().split(), tgt.strip().split()
            if src != tgt:
                alignment = get_alignment(src, tgt)
                alignments.append(alignment)
    return alignments


def get_alignment(source, target):
    """
    Get the visual alignment of two strings.

    Args:
        source (str): Source string.
        target (str): Target string.

    Returns:
        list: containing the strings, their alignment and the edit operations.
    """

    i, j = 0, 0
    alignment = [[] for _ in range(4)]  # 4 lines: source, bars, target, codes
    for code in edit_operations(source, target):
        code = code[0].upper()
        try:
            s = source[i]
        except IndexError:
            pass
        try:
            t = target[j]
        except IndexError:
            pass

        if code == 'D':  # Deletion: empty string on the target side
            t = '*'
        elif code == 'I':  # Insertion: empty string on the source side
            s = '*'
        elif code == 'E':  # Equal: omit the code
            code = ' '

        # Format all elements to the same width.
        width = max(len(x) for x in (s, t, code))
        for line, token in zip(alignment, [s, '|', t, code]):
            line.append(token.center(width))  # pad to width with spaces

        # Increase the counters depending on the operation.
        if code != 'D':  # proceed on the target side, except for deletion
            j += 1
        if code != 'I':  # proceed on the source side, except for insertion
            i += 1
    # Return the accumulated lines.
    return alignment
