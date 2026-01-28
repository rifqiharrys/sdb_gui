import re


def str2bool(v: str) -> bool:
    """
    Transform string True or False to boolean type
    """

    return v in ('True')


def acronym(phrase: str) -> str:
    """
    Generate an acronym from a phrase by taking the first letter of each word
    and converting it to uppercase.
    """

    words = re.split(r'[\s\-]+', phrase)
    return ''.join(word[0].upper() for word in words if word and word[0].isalpha())


def to_title(phrase: str) -> str:
    """
    Convert a variable like phrase to a title case string.
    Change underscores to spaces and capitalize the first letter of each word.
    """

    return phrase.replace('_', ' ').title()