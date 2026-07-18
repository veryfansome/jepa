# Path bootstrap so `unittest discover` works from any cwd: the project root
# (terminal-jepa/) is not an importable package (hyphenated name), so tests add it to
# sys.path explicitly before importing env/datagen.
import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
