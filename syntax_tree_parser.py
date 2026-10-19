from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


EPSILON = "ε"


@dataclass
class TreeNode:
    """One node in a derivation tree."""

    symbol: str
    children: list["TreeNode"] = field(default_factory=list)

    def render(self, prefix: str = "", is_last: bool = True) -> str:
        """Render this node using box-drawing characters."""
        lines = [self.symbol]
        for index, child in enumerate(self.children):
            last = index == len(self.children) - 1
            connector = "└── " if last else "├── "
            child_lines = child.render(
                prefix + ("    " if last else "│   "),
                last,
            ).splitlines()
            lines.append(prefix + connector + child_lines[0])
            lines.extend(child_lines[1:])
        return "\n".join(lines)


@dataclass(frozen=True)
class ParseResult:
    """The result of parsing a sentence."""

    accepted: bool
    tree: TreeNode | None = None
    message: str = ""


class CFGParser:
    """Parse token sequences according to a context-free grammar."""

    def __init__(
        self,
        rules: Mapping[str, str | Iterable[str]],
        start_symbol: str,
    ) -> None:
        if not rules:
            raise ValueError("At least one production rule is required.")
        if start_symbol not in rules:
            raise ValueError(f"Unknown start symbol: {start_symbol!r}.")

        self.start_symbol = start_symbol
        self.nonterminals = set(rules)
        self.productions: dict[str, list[list[str]]] = {
            symbol: self._normalise_productions(value)
            for symbol, value in rules.items()
        }

    @staticmethod
    def _normalise_productions(
        value: str | Iterable[str],
    ) -> list[list[str]]:
        alternatives = [value] if isinstance(value, str) else list(value)
        productions: list[list[str]] = []
        for alternative in alternatives:
            symbols = shlex.split(alternative) if alternative else []
            if symbols and symbols[0].lower() in {"ε", "epsilon"}:
                symbols = []
            productions.append(symbols)
        return productions

    def parse(self, sentence: str | Sequence[str]) -> ParseResult:
        """Return whether *sentence* belongs to the grammar's language."""
        tokens = shlex.split(sentence) if isinstance(sentence, str) else list(sentence)
        memo: dict[tuple[str, int], list[tuple[TreeNode, int]]] = {}
        active: set[tuple[str, int]] = set()

        def parse_symbol(
            symbol: str,
            position: int,
        ) -> list[tuple[TreeNode, int]]:
            key = (symbol, position)
            if key in memo:
                return memo[key]
            if key in active:
                # A left-recursive cycle cannot consume input in this parser.
                return []

            active.add(key)
            matches: list[tuple[TreeNode, int]] = []
            for production in self.productions.get(symbol, [[symbol]]):
                partial: list[tuple[list[TreeNode], int]] = [([], position)]
                for item in production:
                    next_partial: list[tuple[list[TreeNode], int]] = []
                    for children, current_position in partial:
                        if item in self.nonterminals:
                            item_matches = parse_symbol(item, current_position)
                        elif (
                            current_position < len(tokens)
                            and tokens[current_position] == item
                        ):
                            item_matches = [
                                (TreeNode(item), current_position + 1)
                            ]
                        else:
                            item_matches = []
                        for child, end_position in item_matches:
                            next_partial.append(
                                (children + [child], end_position)
                            )
                    partial = next_partial
                    if not partial:
                        break
                matches.extend(
                    (TreeNode(symbol, children), end_position)
                    for children, end_position in partial
                )
            active.remove(key)
            memo[key] = matches
            return matches

        matches = [
            (tree, end)
            for tree, end in parse_symbol(self.start_symbol, 0)
            if end == len(tokens)
        ]
        if matches:
            return ParseResult(True, matches[0][0], "Sentence accepted.")
        return ParseResult(
            False,
            message=(
                "Sentence rejected: it cannot be derived from the grammar "
                f"starting at {self.start_symbol!r}."
            ),
        )


DEFAULT_RULES = {
    "Clause": ["Condition + Consequence"],
    "Condition": ["if Statement"],
    "Consequence": ["then Statement"],
    "Statement": ["subject verb object"],
}


def parse_rule(rule: str) -> tuple[str, str]:
    """Parse a command-line rule such as ``Clause -> Condition + Consequence``."""
    if "->" not in rule:
        raise ValueError(f"Invalid rule {rule!r}; expected LEFT -> RIGHT.")
    left, right = (part.strip() for part in rule.split("->", 1))
    if not left or not right:
        raise ValueError(f"Invalid rule {rule!r}; both sides are required.")
    return left, right


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a sentence and print its CFG derivation tree."
    )
    parser.add_argument(
        "sentence",
        nargs="?",
        default="if subject verb object + then subject verb object",
        help="Sentence to parse (quote it when passing it as one argument).",
    )
    parser.add_argument("--start", default="Clause", help="Start symbol.")
    parser.add_argument(
        "--rule",
        action="append",
        metavar="LEFT->RIGHT",
        help="Production rule; repeat for alternatives.",
    )
    args = parser.parse_args()

    rules: dict[str, list[str]] = {}
    for rule_text in args.rule or []:
        left, right = parse_rule(rule_text)
        rules.setdefault(left, []).append(right)
    if not rules:
        rules = DEFAULT_RULES

    try:
        result = CFGParser(rules, args.start).parse(args.sentence)
    except ValueError as error:
        parser.error(str(error))

    print(result.message)
    if result.tree is not None:
        print("\nDerivation tree:")
        print(result.tree.render())
    raise SystemExit(0 if result.accepted else 1)


if __name__ == "__main__":
    main()