"""Parser: full-obs observation text -> FsState. This is the shared nonprivileged
component that is load-bearing in four places (terminal-jepa.md §5): the parser+BFS
baseline, the planner validity filter, goal-exemplar construction, and (Phase 3) the
belief-state tracker. It must tolerate distractor lines, since those are part of the
observation."""

from . import vocab
from .state import FsState


class ParseError(ValueError):
    pass


def parse_full(text):
    lines = text.split("\n")
    # Banner / dynamic-noise lines precede "cwd:"; skip anything before it.
    try:
        i = next(i for i, l in enumerate(lines) if l.startswith("cwd: "))
    except StopIteration:
        raise ParseError("no 'cwd:' line")
    cwd = vocab.str_to_path(lines[i][len("cwd: "):])
    if i + 1 >= len(lines) or lines[i + 1] != "tree:":
        raise ParseError("no 'tree:' line after cwd")

    dirs, files = set(), {}
    for line in lines[i + 2:]:
        if not line:
            continue
        if line.endswith("/"):
            dirs.add(vocab.str_to_path(line[:-1]))
        else:
            try:
                path_s, class_s = line.rsplit(" [c", 1)
                k = int(class_s.rstrip("]"))
            except ValueError:
                raise ParseError(f"bad tree entry: {line!r}")
            files[vocab.str_to_path(path_s)] = k
    return FsState(dirs, files, cwd).check_invariants()
