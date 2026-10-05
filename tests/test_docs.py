"""Relative links and images in the README and the guides must resolve."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", ROOT / "CHANGELOG.md", *sorted((ROOT / "docs" / "guide").glob("*.md"))]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    headings = re.findall(r"^#+ (.+)$", path.read_text(encoding="utf-8"), flags=re.M)
    return {_slug(h) for h in headings}


def _links(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.S)  # code blocks are not links
    return [t for t in LINK.findall(text) if not re.match(r"[a-z]+:", t)]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(doc):
    assert _links(doc) or doc.name == "CHANGELOG.md"
    for target in _links(doc):
        file_part, _, anchor = target.partition("#")
        resolved = (doc.parent / file_part).resolve() if file_part else doc
        assert resolved.exists(), f"{doc.name}: broken link {target}"
        if anchor and resolved.suffix == ".md":
            assert anchor in _anchors(resolved), f"{doc.name}: missing anchor {target}"


def test_every_guide_is_linked_from_the_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for guide in (ROOT / "docs" / "guide").glob("*.md"):
        assert f"docs/guide/{guide.name}" in readme


def test_slug_rules():
    assert _slug("R and Stata conventions") == "r-and-stata-conventions"
    assert _slug("β-convergence") == "β-convergence"
    assert _slug("What explains club membership?") == "what-explains-club-membership"
