import re
from util.constants import VALID_ARABIC


_whitespace_re = re.compile(r"\s+")


def collapse_whitespace(text):
    text = re.sub(_whitespace_re, " ", text)
    return text


def basic_cleaners(text):
    text = collapse_whitespace(text)
    return text.strip()


def valid_arabic_cleaners(text):
    text = filter(lambda char: char in VALID_ARABIC, text)
    text = collapse_whitespace(''.join(list(text)))
    return text.strip()
