"""HTML cleaning for ASSISTments problem bodies, vendored from the export.

Adapted from the FoundationalASSIST release's ``Code/clean_utils.py``
(Copyright (c) 2024 FoundationalED Authors, licensed CC-BY-NC-4.0 — see
``data/raw/Code/LICENSE``), so the driver no longer imports code out of the
gated data directory. Only the cleaning code moved; the data itself stays
outside the repository as ever. This file inherits the upstream license, which
requires changes to be indicated; ours, previously applied as a runtime patch
(``patch_mathml_parser``) that this vendoring absorbs:

- ``parse_mathml_element`` handles the container elements upstream sent to
  ``get_text()``, which flattens nested structure — an ``<mfrac>`` inside an
  ``<mfenced>`` became ``14`` instead of ``(1/4)``. ``mfenced``, ``mroot``,
  ``mtable``/``mtr`` and ``mspace``-likes are parsed explicitly, and the
  fallback for unknown tags recurses, so nesting always survives.
- Whitespace-only ``<mo>`` (e.g. ``<mo>&#160;</mo>``) renders as a space —
  upstream stripped it to nothing, gluing tokens together ("1 to 12" ->
  "1to12") — except between digits, where some bodies use ``&nbsp;`` to space
  out a numeral and a space would split it.
- Reformatted to this repo's style; upstream's behavior for its own twelve
  MathML tags, and ``clean_problem_body`` / ``parse_mathml_element_children``
  wholesale, are unchanged.

The driver's own extensions that are *not* upstream's concern (ASSISTments
``<ast-r>`` answer blanks, HTML ``<sup>``/``<sub>``) stay in
``processing.make_clean_body``, layered on top.

This module ships with the workspace, never with the ``kcluster`` wheel — the
package is MIT-licensed and this file is not.
"""

import html

import pandas as pd
from bs4 import BeautifulSoup, NavigableString

#: Operators rendered with a space on both sides.
SPACED_OPERATORS = ("÷", "×", "·", "+", "-", "=", "<", ">", "≤", "≥", "≠")


def _children(elem):
    """The element children, skipping the NavigableStrings between them."""
    return [c for c in elem.children if c.name]


def parse_mathml_element(elem):
    """Recursively parse a MathML element to text."""
    if elem.name is None:
        return str(elem).strip()

    # --- container handling added to the upstream parser (see docstring) ---
    if elem.name == "mfenced":  # (a, b) — parentheses were being dropped
        opening, closing = elem.get("open", "("), elem.get("close", ")")
        separators = elem.get("separators", ",")
        parts = [parse_mathml_element(c) for c in _children(elem)]
        if len(parts) <= 1:
            body = "".join(parts)
            # child already supplied parens (mfrac -> "(1/2)"): don't double up
            if opening == "(" and closing == ")" and body.startswith("(") and body.endswith(")"):
                return body
        else:
            out = [parts[0]]
            for idx, part in enumerate(parts[1:]):
                sep = separators[idx] if idx < len(separators) else (separators[-1:] or ",")
                out.append(f"{sep}{part}" if sep.strip() else part)
            body = "".join(out)
        return f"{opening}{body}{closing}"

    if elem.name == "mroot":  # nth root: was rendering ∛27 as "273"
        ch = _children(elem)
        if len(ch) >= 2:
            return f"root{parse_mathml_element(ch[1])}({parse_mathml_element(ch[0])})"

    if elem.name == "mtable":
        return "; ".join(parse_mathml_element(row) for row in _children(elem))
    if elem.name == "mtr":
        return ", ".join(parse_mathml_element(cell) for cell in _children(elem))
    if elem.name in ("mspace", "msline", "none"):
        return " "

    # Whitespace-only operator: a space, unless it sits inside a spaced-out
    # numeral ("0. 2") where a space would split the number
    if elem.name == "mo" and not elem.get_text(strip=True):
        def in_number(sibling):
            if sibling is None:
                return False
            return sibling.name == "mn" or (sibling.name == "mo" and sibling.get_text(strip=True) == ".")

        if in_number(elem.find_previous_sibling()) and in_number(elem.find_next_sibling()):
            return ""
        return " "

    # --- the tags upstream handles, behavior unchanged ---
    if elem.name == "mfrac":
        children = _children(elem)
        if len(children) >= 2:
            return f"({parse_mathml_element(children[0])}/{parse_mathml_element(children[1])})"
        return elem.get_text(strip=True)

    if elem.name == "msup":
        children = _children(elem)
        if len(children) >= 2:
            return f"{parse_mathml_element(children[0])}^{parse_mathml_element(children[1])}"
        return elem.get_text(strip=True)

    if elem.name == "msub":
        children = _children(elem)
        if len(children) >= 2:
            return f"{parse_mathml_element(children[0])}_{parse_mathml_element(children[1])}"
        return elem.get_text(strip=True)

    if elem.name == "msqrt":
        return f"√({parse_mathml_element_children(elem)})"

    if elem.name == "mo":
        op = elem.get_text(strip=True)
        return f" {op} " if op in SPACED_OPERATORS else op

    if elem.name in ("mn", "mi", "mtext"):
        return elem.get_text(strip=True)

    if elem.name in ("mrow", "math", "mpadded", "mstyle"):
        return parse_mathml_element_children(elem)

    # mtd / menclose / mover / munder / mstack / mlongdiv / msrow / msgroup and
    # anything unknown: recurse so nested structure is preserved (upstream
    # flattened these with get_text())
    return parse_mathml_element_children(elem)


def parse_mathml_element_children(elem):
    """Parse all children of a MathML element."""
    parts = []
    for child in elem.children:
        if isinstance(child, NavigableString):
            if text := str(child).strip():
                parts.append(text)
        elif child.name:
            parts.append(parse_mathml_element(child))
    return "".join(parts)


def clean_problem_body(text):
    """
    Clean HTML problem body with full MathML handling.

    Handles:
    - Inline MathML (<math>, <mfrac>, <msup>, etc.) → (4/3), x^2
    - Wiris math images (data-mathml attribute) → [15÷12]
    - Tables → [Table: Col1 | Col2 ...]
    - Regular images → [image]
    - HTML entities → decoded properly
    """
    if pd.isna(text) or text == "":
        return ""
    soup = BeautifulSoup(str(text), "html.parser")

    # 1. Handle inline MathML
    for math in soup.find_all("math"):
        parsed = parse_mathml_element(math)
        math.replace_with(f" {parsed} ")

    # 2. Handle Wiris images
    for img in soup.find_all("img"):
        alt = img.get("alt", "")
        src = img.get("src", "")
        data_mathml = img.get("data-mathml", "")

        if "wiris" in src.lower() or "pluginwiris" in src:
            if alt and alt.strip() and alt not in ["NO ALT", "NONE"]:
                img.replace_with(f" [{alt.strip()}] ")
            elif data_mathml:
                math_str = data_mathml.replace("«", "<").replace("»", ">").replace("¨", '"')
                msoup = BeautifulSoup(math_str, "html.parser")
                math_elem = msoup.find("math")
                if math_elem:
                    mtext = parse_mathml_element(math_elem)
                else:
                    mtext = msoup.get_text(separator="")
                mtext = mtext.replace("§#247;", "÷").replace("§#215;", "×")
                mtext = mtext.replace("§#8722;", "-").replace("§#160;", " ").replace("§#183;", "·")
                mtext = mtext.replace("§#", "&#")
                mtext = html.unescape(mtext).strip()
                img.replace_with(f" [{mtext}] " if mtext else " [math] ")
            else:
                img.replace_with(" [math] ")
        elif alt and alt.strip():
            img.replace_with(f" [Image: {alt.strip()[:100]}] ")
        else:
            img.replace_with(" [image] ")

    # 3. Handle tables
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            table.replace_with(f"\n[Table:\n{chr(10).join(rows)}]\n")
        else:
            table.decompose()

    text = soup.get_text(separator=" ")
    text = html.unescape(text)
    text = " ".join(text.split())
    return text.strip()
