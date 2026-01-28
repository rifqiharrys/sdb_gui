import re
import sys
from pathlib import Path


def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource.
    Works for both development and PyInstaller.
    Finds project root regardless of where function is called from.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = Path(sys._MEIPASS) # type: ignore
    except AttributeError:
        # Find project root by going up from sdb/ directory
        base_path = Path(__file__).parent.parent.resolve()
    
    return str(base_path / relative_path)

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