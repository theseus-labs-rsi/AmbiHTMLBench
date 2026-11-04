from __future__ import annotations

import argparse
from pathlib import Path


BLOCK_MATCHING = r"""\subsubsection{Block Matching Algorithm}

Given a ground-truth object set \(G\) and a predicted object set \(P\), the evaluator first brings both sides into the same rendered-block representation. Tables are matched as table objects. Formula markup is converted to ordinary text after removing only dollar delimiters, so mathematical expressions contribute to page text fidelity rather than forming a separate metric. Regular text, formula text, and cells from unmatched tables are then placed in a shared text pool. This design gives text credit when a model expresses a table as ordinary HTML text, while reserving table-structure credit for predictions that preserve table objects.

Table candidates are scored by a fixed combination of table content, table structure, text agreement, and geometric agreement. The candidate matrix is solved with a thresholded Hungarian assignment, so each table can be used at most once and the assignment is global rather than greedy.

Text requires a more flexible alignment because generated HTML often merges adjacent annotations or splits one annotation into multiple DOM elements. We therefore construct a bipartite graph between ground-truth and predicted text blocks using text similarity as the primary evidence. Connected components of this graph define local candidate regions that may contain one-to-one, one-to-many, many-to-one, or many-to-many correspondences. Within each region, the evaluator searches over a small set of reading, horizontal, graph-guided, and locally reordered block orders. For a fixed pair of orders, a monotone segmentation dynamic program partitions the two sequences into matched groups. These groups are subsequently used both for page-level text scoring and for layout graph construction. Thus the text and layout measurements are based on a shared alignment rather than on independent heuristics. Implementation details are deferred to Appendix~\ref{app:objective-details}."""


CONTENT = r"""\subsubsection{Content consistency}

For each matched table, TEDS measures agreement in both cell structure and cell content, whereas TEDS-S suppresses cell text and measures structure alone. The implementation parses tables into ordered rows and cells and applies a normalized row/cell edit approximation:
\[
S_{\mathrm{table}}=1-\frac{d_{\mathrm{row/cell}}}
{\max(|T_g|,|T_p|,1)}.
\]
This representation preserves row and cell boundaries, including the row/cell structure exposed by merged cells, while avoiding sensitivity to irrelevant HTML serialization. We report these metrics only over the 263 pages containing ground-truth tables.

Text fidelity is computed once per page rather than averaged over local matches. After the final text alignment is obtained, the matched groups are ordered at page level, the ground-truth text in the groups is concatenated, and the predicted text in the corresponding groups is concatenated. The score is the normalized character edit distance
\[
D_{\mathrm{text}} =
\frac{\operatorname{Lev}(T_g,T_p)}
{\max(|T_g|,|T_p|,1)} .
\]
Lower is better. Text preprocessing removes whitespace and dollar delimiters but preserves punctuation, mathematical symbols, and multilingual characters. The ground-truth symbol \(\xi\) is treated as a wildcard during comparison.

The objective output therefore separates table content, table structure, and page text. Pages without ground-truth tables are marked not applicable for TEDS and TEDS-S rather than being assigned artificial zero or perfect values."""


APP_TEXT = r"""\subsection{Text normalization and grouping}

Text matching uses a conservative normalization. Whitespace and dollar delimiters are removed, while punctuation, mathematical symbols, CJK characters, Japanese, Korean, and other Unicode letters are retained. A ground-truth \(\xi\) is interpreted as a wildcard character during edit-distance computation. Placeholder elements are removed before this stage.

For every ground-truth and predicted text pair, the evaluator computes \texttt{partial\_ratio} after the above normalization. A bipartite edge is created when
\[
s_{\mathrm{partial}} \ge 0.66 .
\]
Unlike earlier overlap-driven variants, this candidate graph is text-led: geometry is used later for layout scoring rather than for admitting text edges. Connected components of the bipartite graph form local candidate regions. Singleton blocks without edges are kept as residual insertions or deletions, so they still affect page text distance but do not create layout nodes.

For each connected component, the evaluator constructs several deterministic order hypotheses: visual reading order, horizontal order, reversed reading order, and a graph-guided order that places a block near its strongest partners on the other side. For components whose total number of blocks is small enough, the evaluator additionally tries bounded local moves, including short-range insertions and reversals. A candidate order is accepted only when it lowers the component-level concatenated edit distance.

Given fixed ground-truth and predicted orders, the evaluator computes the final groups with a monotone segmentation dynamic program. Each transition may consume a short ground-truth segment, a short predicted segment, or both. Two-sided transitions are allowed only if all blocks in the segment are supported by at least one bipartite edge inside the segment. One-sided transitions represent insertions or deletions and receive full edit cost. The recovered groups therefore support one-to-one, one-to-many, many-to-one, and many-to-many matches while preventing unsupported distant text from being grouped solely to reduce local cost.

After all components and residual blocks are processed, the evaluator performs one page-level ordering pass over the recovered groups and computes a single normalized edit distance:
\[
D_{\mathrm{text}}=
\frac{\operatorname{Lev}(T_g,T_p)}
{\max(|T_g|,|T_p|,1)}.
\]
This max denominator makes the score symmetric with respect to which side contains more characters. A lower score is better. The same recovered groups, excluding groups with an empty side, are passed to the layout scorer."""


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[:i] + replacement + "\n\n" + text[j:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Update objective evaluation sections in a LaTeX paper.")
    parser.add_argument("tex", type=Path, help="Path to the LaTeX source to update")
    args = parser.parse_args()
    text = args.tex.read_text(encoding="utf-8")
    text = replace_between(text, r"\subsubsection{Block Matching Algorithm}", r"\subsubsection{Content consistency}", BLOCK_MATCHING)
    text = replace_between(text, r"\subsubsection{Content consistency}", r"\subsubsection{Layout consistency}", CONTENT)
    text = replace_between(text, r"\subsection{Text normalization and grouping}", r"\subsection{Layout graph similarity}", APP_TEXT)
    args.tex.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
