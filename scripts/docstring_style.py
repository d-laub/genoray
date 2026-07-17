"""Classify every docstring in the package as numpy-, google-, or plain-style.

This is the *oracle* for the numpy -> google docstring migration: it walks the
package with :mod:`ast`, inspects each module/class/function docstring, and labels
it so agents can be pointed at exactly the docstrings that still need converting.

``plain`` means "has a docstring but no recognized sections" (a bare summary, an
attribute doc, a ``.. note::``-only body) -- already google-compatible, nothing to
convert. ``numpy`` (underlined section headers) is the work queue; ``google``
(``Args:``-style headers) is done. A docstring mixing both counts as ``numpy``.

    # summary counts + per-file breakdown
    pixi run python scripts/docstring_style.py

    # the work queue: file:line: qualname for every numpy docstring
    pixi run python scripts/docstring_style.py --list numpy

Exit status is non-zero while any ``numpy`` docstring remains, so the same command
doubles as a migration-complete gate.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Napoleon section titles (case-insensitive), the union of the numpy and google
# vocabularies. https://sphinxcontrib-napoleon.readthedocs.io/en/latest/
_SECTION_TITLES = frozenset(
    {
        "args",
        "arguments",
        "attention",
        "attributes",
        "caution",
        "danger",
        "error",
        "example",
        "examples",
        "hint",
        "important",
        "keyword args",
        "keyword arguments",
        "methods",
        "note",
        "notes",
        "other parameters",
        "parameters",
        "raise",
        "raises",
        "receive",
        "receives",
        "return",
        "returns",
        "references",
        "see also",
        "tip",
        "todo",
        "warning",
        "warnings",
        "warn",
        "warns",
        "yield",
        "yields",
    }
)


class Style(str, Enum):
    """Docstring section style."""

    NUMPY = "numpy"
    GOOGLE = "google"
    PLAIN = "plain"


@dataclass(frozen=True)
class Docstring:
    """A located docstring and its classified style."""

    file: Path
    lineno: int
    qualname: str
    style: Style


def _is_underline(line: str) -> bool:
    """True for a numpy section underline: a run of ``-`` (optionally indented)."""
    s = line.strip()
    return len(s) >= 3 and set(s) == {"-"}


def classify(docstring: str) -> Style:
    """Classify a (cleaned) docstring as numpy, google, or plain.

    Numpy wins over google when both appear, so a half-converted docstring stays in
    the work queue.

    Args:
        docstring: The docstring body, already dedented (e.g. via
            :func:`ast.get_docstring` with ``clean=True``).

    Returns:
        The detected :class:`Style`.
    """
    lines = docstring.splitlines()
    google = False
    for i, raw in enumerate(lines):
        line = raw.strip()
        # numpy: a section title on its own line, underlined by dashes below it.
        if line.lower() in _SECTION_TITLES:
            if i + 1 < len(lines) and _is_underline(lines[i + 1]):
                return Style.NUMPY
        # google: a section title immediately followed by a colon, e.g. ``Args:``.
        if line.endswith(":") and line[:-1].strip().lower() in _SECTION_TITLES:
            google = True
    return Style.GOOGLE if google else Style.PLAIN


def _qualname(stack: list[str], node: ast.AST) -> str:
    """Dotted qualified name for a definition node, given its enclosing name stack."""
    name = getattr(node, "name", None)
    if name is None:  # Module
        return "<module>"
    return ".".join([*stack, name]) if stack else name


def scan_file(path: Path) -> list[Docstring]:
    """Classify every docstring in one Python file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[Docstring] = []

    def visit(node: ast.AST, stack: list[str]) -> None:
        doc = ast.get_docstring(node, clean=True)
        if doc is not None:
            # get_docstring returns the summary at the node's own lineno for
            # module; for def/class the string is the first stmt in the body.
            body = getattr(node, "body", [])
            lineno = body[0].lineno if body else getattr(node, "lineno", 1)
            found.append(Docstring(path, lineno, _qualname(stack, node), classify(doc)))
        child_stack = stack
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            child_stack = [*stack, node.name]
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                visit(child, child_stack)

    visit(tree, [])
    return found


def scan(source: Path) -> list[Docstring]:
    """Classify every docstring under a file or directory (recursively)."""
    files = [source] if source.is_file() else sorted(source.rglob("*.py"))
    out: list[Docstring] = []
    for f in files:
        out.extend(scan_file(f))
    return out


def main() -> int:
    """CLI entry point. Returns a process exit status (non-zero if numpy remains)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "source",
        nargs="?",
        default="python/genoray",
        type=Path,
        help="File or directory to scan (default: python/genoray).",
    )
    parser.add_argument(
        "--list",
        choices=[s.value for s in Style],
        help="List file:line: qualname for docstrings of this style, then exit.",
    )
    args = parser.parse_args()

    docs = scan(args.source)

    if args.list is not None:
        for d in docs:
            if d.style.value == args.list:
                print(f"{d.file}:{d.lineno}: {d.qualname}")
        return 0

    counts = {s: 0 for s in Style}
    per_file: dict[Path, dict[Style, int]] = {}
    for d in docs:
        counts[d.style] += 1
        per_file.setdefault(d.file, {s: 0 for s in Style})[d.style] += 1

    print(f"{'file':<45} {'numpy':>6} {'google':>7} {'plain':>6}")
    print("-" * 66)
    for f in sorted(per_file, key=lambda p: -per_file[p][Style.NUMPY]):
        c = per_file[f]
        if c[Style.NUMPY] or c[Style.GOOGLE]:
            rel = f.as_posix().replace("python/genoray/", "")
            print(
                f"{rel:<45} {c[Style.NUMPY]:>6} {c[Style.GOOGLE]:>7} {c[Style.PLAIN]:>6}"
            )
    print("-" * 66)
    total = sum(counts.values())
    print(
        f"{'TOTAL (' + str(total) + ' docstrings)':<45} "
        f"{counts[Style.NUMPY]:>6} {counts[Style.GOOGLE]:>7} {counts[Style.PLAIN]:>6}"
    )
    return 1 if counts[Style.NUMPY] else 0


if __name__ == "__main__":
    raise SystemExit(main())
