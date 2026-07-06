"""Thin presentation helpers for the Quarto example notebooks.

These functions only *render* a typed :class:`~schenberg.core.graph.Formula` or
:class:`~schenberg.core.graph.FormulaGraph` — its LaTeX (derived from the
symbolic IR, never hand-written) and its Mermaid dependency diagram — into the
Markdown that Quarto turns into MathJax and Mermaid.
No pricing maths lives here; instrument formulas stay in each notebook's graph.

Call them from a cell tagged ``#| output: asis`` so the printed Markdown is
processed by Quarto rather than shown verbatim::

    #| output: asis
    show_formulas(graph)
    show_graph(graph)
"""

from __future__ import annotations

from typing import Any


def show_formulas(graph: Any, *, title: str = "Formulas") -> None:
    """Emit every term as ``$$ symbol = latex $$`` (rendered by MathJax)."""
    if title:
        print(f"**{title}**\n")
    for latex in graph.formulas().values():
        print(f"$$ {latex} $$\n")


def show_formula(graph: Any, term: str) -> None:
    """Emit a single term's LaTeX as a display equation."""
    print(f"$$ {graph.formula_of(term)} $$\n")


def show_graph(graph: Any) -> None:
    """Emit the dependency DAG as a Mermaid flowchart block."""
    print("```{mermaid}")
    print(graph.to_mermaid())
    print("```\n")


def show_explain(graph: Any, *, view: str) -> None:
    """Emit ``graph.explain(view=...)`` inside a fenced text block."""
    print("```text")
    print(graph.explain(view=view))
    print("```\n")
