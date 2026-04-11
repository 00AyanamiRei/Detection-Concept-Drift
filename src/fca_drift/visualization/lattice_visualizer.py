# """
# lattice_visualizer.py
# ─────────────────────────────────────────────────
# Генерує PNG-граф концептуальної решітки (до і після дрифту)
# і повертає base64-рядок для вбудовування в HTML-звіт.
# """

# import io
# import base64
# import os
# import math
# from pathlib import Path
# from typing import FrozenSet, Set, Dict, List, Optional, Tuple

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
# import matplotlib.patheffects as pe
# import networkx as nx
# import numpy as np


# # ─── Кольорова схема ──────────────────────────────────────────────────────────
# COLOR = {
#     "stable":  "#4A9ECA",   # синій  — концепт існує в обох решітках
#     "lost":    "#E05C5C",   # червоний — концепт зник після дрифту
#     "gained":  "#5BAD7A",   # зелений  — концепт з'явився після дрифту
#     "top":     "#8B8B8B",   # сірий    — TOP/BOTTOM
#     "edge_stable": "#AAAAAA",
#     "edge_lost":   "#E05C5C",
#     "edge_gained": "#5BAD7A",
#     "bg":      "#FAFAFA",
#     "title_before": "#1A6EA0",
#     "title_after":  "#B03030",
# }


# class ConceptLatticeRenderer:
#     """
#     Будує часткове впорядкування концептів з набору інтенсій (frozenset-ів).
#     Ребра: C_i → C_j якщо intent(i) ⊂ intent(j) і немає C_k між ними.
#     """

#     def __init__(self, intents: Set[FrozenSet]):
#         # Додаємо TOP (порожня множина) і BOTTOM (всі атрибути)
#         all_attrs: FrozenSet = frozenset().union(*intents) if intents else frozenset()
#         self.nodes: List[FrozenSet] = sorted(
#             intents | {frozenset(), all_attrs},
#             key=lambda x: len(x)
#         )
#         self.edges: List[Tuple[int, int]] = self._build_cover_relations()

#     def _build_cover_relations(self) -> List[Tuple[int, int]]:
#         """Пряме покриття: i < j і немає k такого що i < k < j."""
#         n = self.nodes
#         edges = []
#         for i, a in enumerate(n):
#             for j, b in enumerate(n):
#                 if i == j:
#                     continue
#                 if not a.issubset(b) or a == b:
#                     continue
#                 # перевіряємо що немає проміжного елементу
#                 covered = False
#                 for k, c in enumerate(n):
#                     if k == i or k == j:
#                         continue
#                     if a.issubset(c) and c.issubset(b) and c != a and c != b:
#                         covered = True
#                         break
#                 if not covered:
#                     edges.append((i, j))
#         return edges

#     def to_networkx(self) -> nx.DiGraph:
#         G = nx.DiGraph()
#         for i, intent in enumerate(self.nodes):
#             label = self._format_intent(intent, i)
#             G.add_node(i, intent=intent, label=label)
#         for i, j in self.edges:
#             G.add_edge(i, j)
#         return G

#     @staticmethod
#     def _format_intent(intent: FrozenSet, idx: int) -> str:
#         if len(intent) == 0:
#             return "TOP\n∅"
#         attrs = sorted(str(a) for a in intent)
#         if len(attrs) <= 3:
#             return "{" + ", ".join(attrs) + "}"
#         return "{" + ", ".join(attrs[:3]) + "…}"


# def _hierarchical_layout(G: nx.DiGraph) -> Dict[int, Tuple[float, float]]:
#     """
#     Розставляє вузли по рівнях (за розміром інтенсії).
#     Вузли одного рівня — рівномірно по горизонталі.
#     """
#     levels: Dict[int, List[int]] = {}
#     for node in G.nodes:
#         lv = len(G.nodes[node]["intent"])
#         levels.setdefault(lv, []).append(node)

#     max_width = max(len(v) for v in levels.values()) if levels else 1
#     pos = {}
#     sorted_levels = sorted(levels.keys())
#     for rank, lv in enumerate(sorted_levels):
#         nodes_at_lv = levels[lv]
#         n = len(nodes_at_lv)
#         for i, node in enumerate(sorted(nodes_at_lv)):
#             x = (i + 1) / (n + 1) * max_width
#             y = rank
#             pos[node] = (x, y)
#     return pos


# def _classify_nodes(
#     before_intents: Set[FrozenSet],
#     after_intents: Set[FrozenSet],
# ) -> Tuple[Set[FrozenSet], Set[FrozenSet], Set[FrozenSet]]:
#     stable = before_intents & after_intents
#     lost   = before_intents - after_intents
#     gained = after_intents  - before_intents
#     return stable, lost, gained


# class LatticeVisualizer:
#     """Візуалізує решітку концептів до та після дрифту."""

#     def __init__(self, dpi: int = 150, figsize: Tuple[float, float] = (14, 6)):
#         self.dpi = dpi
#         self.figsize = figsize

#     def render_comparison_png(
#         self,
#         intents_before: Set[FrozenSet],
#         intents_after:  Set[FrozenSet],
#         episode_id:     int = 0,
#         delta_lt:       float = 0.0,
#         output_dir:     Optional[str] = None,
#     ) -> str:
#         """
#         Малює side-by-side: решітка ДО і ПІСЛЯ дрифту.
#         Повертає base64 PNG для вбудовування в HTML.
#         Якщо output_dir вказано — також зберігає файл.
#         """
#         stable, lost, gained = _classify_nodes(intents_before, after_intents=intents_after)

#         lat_before = ConceptLatticeRenderer(intents_before)
#         lat_after  = ConceptLatticeRenderer(intents_after)

#         G_before = lat_before.to_networkx()
#         G_after  = lat_after.to_networkx()

#         pos_before = _hierarchical_layout(G_before)
#         pos_after  = _hierarchical_layout(G_after)

#         fig, axes = plt.subplots(1, 2, figsize=self.figsize, facecolor=COLOR["bg"])
#         fig.suptitle(
#             f"Lattice comparison — Episode #{episode_id}   |   ΔL = {delta_lt:.4f}",
#             fontsize=13, fontweight="bold", color="#333333", y=1.01
#         )

#         self._draw_lattice(
#             ax=axes[0],
#             G=G_before,
#             pos=pos_before,
#             lost_intents=lost,
#             gained_intents=set(),
#             title=f"Window t−1  (before drift)",
#             title_color=COLOR["title_before"],
#             mode="before",
#         )
#         self._draw_lattice(
#             ax=axes[1],
#             G=G_after,
#             pos=pos_after,
#             lost_intents=set(),
#             gained_intents=gained,
#             title=f"Window t  (after drift)",
#             title_color=COLOR["title_after"],
#             mode="after",
#         )

#         # Легенда
#         legend_patches = [
#             mpatches.Patch(color=COLOR["stable"],  label="Stable concept"),
#             mpatches.Patch(color=COLOR["lost"],    label="Lost concept"),
#             mpatches.Patch(color=COLOR["gained"],  label="New concept"),
#         ]
#         fig.legend(
#             handles=legend_patches,
#             loc="lower center",
#             ncol=3,
#             fontsize=9,
#             framealpha=0.0,
#             bbox_to_anchor=(0.5, -0.04),
#         )

#         # Stats strip
#         n_lost   = len(lost)
#         n_gained = len(gained)
#         n_stable = len(stable)
#         sim = len(stable) / max(len(intents_before | intents_after), 1)
#         fig.text(
#             0.5, -0.09,
#             f"Intents: before={len(intents_before)}  after={len(intents_after)}  "
#             f"stable={n_stable}  lost={n_lost}  gained={n_gained}   "
#             f"Jaccard similarity={sim:.3f}",
#             ha="center", fontsize=9, color="#666666",
#         )

#         plt.tight_layout()

#         # Зберігаємо в файл якщо потрібно
#         if output_dir:
#             Path(output_dir).mkdir(parents=True, exist_ok=True)
#             fpath = os.path.join(output_dir, f"lattice_episode_{episode_id:02d}.png")
#             fig.savefig(fpath, dpi=self.dpi, bbox_inches="tight",
#                         facecolor=COLOR["bg"])

#         # Base64 для HTML
#         buf = io.BytesIO()
#         fig.savefig(buf, format="png", dpi=self.dpi,
#                     bbox_inches="tight", facecolor=COLOR["bg"])
#         plt.close(fig)
#         buf.seek(0)
#         return base64.b64encode(buf.read()).decode("utf-8")

#     def _draw_lattice(
#         self,
#         ax,
#         G: nx.DiGraph,
#         pos: Dict,
#         lost_intents:   Set[FrozenSet],
#         gained_intents: Set[FrozenSet],
#         title: str,
#         title_color: str,
#         mode: str,  # "before" | "after"
#     ):
#         ax.set_facecolor(COLOR["bg"])
#         ax.set_title(title, fontsize=11, color=title_color, pad=8)
#         ax.axis("off")

#         if len(G.nodes) == 0:
#             ax.text(0.5, 0.5, "Empty lattice", ha="center", va="center",
#                     transform=ax.transAxes, color="#999")
#             return

#         # Кольори вузлів
#         node_colors = []
#         node_edge_colors = []
#         for n in G.nodes:
#             intent = G.nodes[n]["intent"]
#             if intent in lost_intents:
#                 node_colors.append(COLOR["lost"])
#                 node_edge_colors.append("#A03030")
#             elif intent in gained_intents:
#                 node_colors.append(COLOR["gained"])
#                 node_edge_colors.append("#2A7A4A")
#             elif len(intent) == 0:
#                 node_colors.append(COLOR["top"])
#                 node_edge_colors.append("#555555")
#             else:
#                 node_colors.append(COLOR["stable"])
#                 node_edge_colors.append("#2060A0")

#         # Кольори ребер
#         edge_colors = []
#         for u, v in G.edges:
#             iu = G.nodes[u]["intent"]
#             iv = G.nodes[v]["intent"]
#             if iu in lost_intents or iv in lost_intents:
#                 edge_colors.append(COLOR["edge_lost"])
#             elif iu in gained_intents or iv in gained_intents:
#                 edge_colors.append(COLOR["edge_gained"])
#             else:
#                 edge_colors.append(COLOR["edge_stable"])

#         nx.draw_networkx_edges(
#             G, pos, ax=ax,
#             edge_color=edge_colors,
#             arrows=True,
#             arrowstyle="-|>",
#             arrowsize=12,
#             width=1.2,
#             alpha=0.7,
#             connectionstyle="arc3,rad=0.05",
#         )

#         nx.draw_networkx_nodes(
#             G, pos, ax=ax,
#             node_color=node_colors,
#             node_size=900,
#             linewidths=1.5,
#             edgecolors=node_edge_colors,
#         )

#         labels = {n: G.nodes[n]["label"] for n in G.nodes}
#         nx.draw_networkx_labels(
#             G, pos, labels=labels, ax=ax,
#             font_size=7,
#             font_color="white",
#             font_weight="bold",
#         )
"""
lattice_visualizer.py
─────────────────────────────────────────────────
Генерує PNG-граф концептуальної решітки (до і після дрифту)
і повертає base64-рядок для вбудовування в HTML-звіт.

ВИКОРИСТАННЯ в твоєму проекті:
    from lattice_visualizer import LatticeVisualizer

    viz = LatticeVisualizer()

    # під час детекції — зберігай пари решіток
    # (викликати одразу після DriftAggregatorV2)
    for episode in merged_episodes:
        snapshot = lattice_snapshots.get(episode.start)
        if snapshot:
            b64 = viz.render_comparison_png(
                intents_before = snapshot["before"],
                intents_after  = snapshot["after"],
                episode_id     = episode.index,
                delta_lt       = episode.dlt_max,
                output_dir     = output_dir,
            )
            episode.lattice_png_b64 = b64   # зберігаємо в епізод

    # в HTML-звіті:
    # <img src="data:image/png;base64,{episode.lattice_png_b64}">
"""

import io
import base64
import os
import math
from pathlib import Path
from typing import FrozenSet, Set, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np


# ─── Кольорова схема ──────────────────────────────────────────────────────────
COLOR = {
    "stable":  "#4A9ECA",   # синій  — концепт існує в обох решітках
    "lost":    "#E05C5C",   # червоний — концепт зник після дрифту
    "gained":  "#5BAD7A",   # зелений  — концепт з'явився після дрифту
    "top":     "#8B8B8B",   # сірий    — TOP (∅)
    "bottom":  "#5C5C8B",   # темно-синій/сірий — BOTTOM (всі атрибути)
    "edge_stable": "#AAAAAA",
    "edge_lost":   "#E05C5C",
    "edge_gained": "#5BAD7A",
    "bg":      "#FAFAFA",
    "title_before": "#1A6EA0",
    "title_after":  "#B03030",
}


class ConceptLattice:
    """
    Будує часткове впорядкування концептів з набору інтенсій (frozenset-ів).
    Ребра: C_i → C_j якщо intent(i) ⊂ intent(j) і немає C_k між ними.
    """

    def __init__(self, intents: Set[FrozenSet]):
        # Додаємо TOP (порожня множина) і BOTTOM (всі атрибути)
        all_attrs: FrozenSet = frozenset().union(*intents) if intents else frozenset()
        self.nodes: List[FrozenSet] = sorted(
            intents | {frozenset(), all_attrs},
            key=lambda x: len(x)
        )
        self.edges: List[Tuple[int, int]] = self._build_cover_relations()

    def _build_cover_relations(self) -> List[Tuple[int, int]]:
        """Пряме покриття: i < j і немає k такого що i < k < j."""
        n = self.nodes
        edges = []
        for i, a in enumerate(n):
            for j, b in enumerate(n):
                if i == j:
                    continue
                if not a.issubset(b) or a == b:
                    continue
                # перевіряємо що немає проміжного елементу
                covered = False
                for k, c in enumerate(n):
                    if k == i or k == j:
                        continue
                    if a.issubset(c) and c.issubset(b) and c != a and c != b:
                        covered = True
                        break
                if not covered:
                    edges.append((i, j))
        return edges

    def to_networkx(self) -> nx.DiGraph:
        G = nx.DiGraph()
        # Знаходимо розмір BOTTOM вузла (всі атрибути)
        max_intent_size = max(len(n) for n in self.nodes) if self.nodes else 0
        for i, intent in enumerate(self.nodes):
            label = self._format_intent(intent, i, max_intent_size)
            G.add_node(i, intent=intent, label=label)
        for i, j in self.edges:
            G.add_edge(i, j)
        return G

    @staticmethod
    def _format_intent(intent: FrozenSet, idx: int, max_intent_size: int = 0) -> str:
        """Форматує intent для відображення на вузлі.

        - TOP (∅)           → "TOP\n∅"
        - BOTTOM (всі атрибути) → "BOT\n{N}"
        - ≤4 attrs          → повний список:  "{0, 2, 4}"
        - >4 attrs          → перші 3 + залишок: "{0,2,3\n…+5}"
        """
        if len(intent) == 0:
            return "TOP\n∅"
        attrs = sorted(int(a) if str(a).isdigit() else a for a in intent)
        # BOTTOM — вузол з максимальною кількістю атрибутів
        if max_intent_size > 0 and len(attrs) == max_intent_size:
            return f"BOT\n{{{len(attrs)}}}"
        if len(attrs) <= 4:
            return "{" + ", ".join(str(a) for a in attrs) + "}"
        # Показуємо перші 3 атрибути + кількість прихованих
        shown = ", ".join(str(a) for a in attrs[:3])
        hidden = len(attrs) - 3
        return "{" + shown + "\n…+" + str(hidden) + "}"


def _hierarchical_layout(G: nx.DiGraph, max_width: Optional[int] = None) -> Dict[int, Tuple[float, float]]:
    """
    Розставляє вузли по рівнях (за розміром інтенсії).

    Конвенція діаграми Хассе (FCA):
      - TOP (∅, len=0)          → ЗНИЗУ  (y=0)
      - BOTTOM (всі атрибути)   → ЗВЕРХУ (y=max_rank)

    Тобто y зростає зі збільшенням len(intent).
    matplotlib малює y=0 знизу → результат правильний без інверсії осі.
    """
    levels: Dict[int, List[int]] = {}
    for node in G.nodes:
        lv = len(G.nodes[node]["intent"])
        levels.setdefault(lv, []).append(node)

    if max_width is None:
        max_width = max(len(v) for v in levels.values()) if levels else 1

    # sorted ascending: len=0 (TOP) → rank=0 (низ), len=max (BOTTOM) → rank=N (верх)
    sorted_levels = sorted(levels.keys())
    n_levels = len(sorted_levels)

    pos = {}
    for rank, lv in enumerate(sorted_levels):
        nodes_at_lv = levels[lv]
        n = len(nodes_at_lv)
        for i, node in enumerate(sorted(nodes_at_lv)):
            x = (i + 1) / (n + 1) * max_width
            y = rank          # rank=0 → знизу, rank=N → зверху
            pos[node] = (x, y)
    return pos


def _classify_nodes(
    before_intents: Set[FrozenSet],
    after_intents: Set[FrozenSet],
) -> Tuple[Set[FrozenSet], Set[FrozenSet], Set[FrozenSet]]:
    stable = before_intents & after_intents
    lost   = before_intents - after_intents
    gained = after_intents  - before_intents
    return stable, lost, gained


class LatticeVisualizer:

    def __init__(self, dpi: int = 150, figsize: Tuple[float, float] = (14, 6)):
        self.dpi = dpi
        self.figsize = figsize

    def render_comparison_png(
        self,
        intents_before: Set[FrozenSet],
        intents_after:  Set[FrozenSet],
        episode_id:     int = 0,
        delta_lt:       float = 0.0,
        output_dir:     Optional[str] = None,
    ) -> str:
        """
        Малює side-by-side: решітка ДО і ПІСЛЯ дрифту.
        Повертає base64 PNG для вбудовування в HTML.
        Якщо output_dir вказано — також зберігає файл.
        """
        stable, lost, gained = _classify_nodes(intents_before, after_intents=intents_after)

        lat_before = ConceptLattice(intents_before)
        lat_after  = ConceptLattice(intents_after)

        G_before = lat_before.to_networkx()
        G_after  = lat_after.to_networkx()

        # Обчислити спільну ширину для синхронізації вирівнювання
        # Це залишає вузли на одній висоті для обох графів
        levels_before: Dict[int, List[int]] = {}
        for node in G_before.nodes:
            lv = len(G_before.nodes[node]["intent"])
            levels_before.setdefault(lv, []).append(node)

        levels_after: Dict[int, List[int]] = {}
        for node in G_after.nodes:
            lv = len(G_after.nodes[node]["intent"])
            levels_after.setdefault(lv, []).append(node)

        # max_width - це максимум кількості вузлів на рівні
        max_width_before = max(len(v) for v in levels_before.values()) if levels_before else 1
        max_width_after = max(len(v) for v in levels_after.values()) if levels_after else 1
        shared_max_width = max(max_width_before, max_width_after)

        pos_before = _hierarchical_layout(G_before, shared_max_width)
        pos_after  = _hierarchical_layout(G_after, shared_max_width)

        # Динамічний figsize на основі кількості концептів
        num_concepts = max(len(intents_before), len(intents_after))
        height = max(4.0, num_concepts * 0.8)
        figsize = (13, height)

        fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=COLOR["bg"])
        fig.suptitle(
            f"Lattice comparison — Episode #{episode_id}   |   ΔL = {delta_lt:.4f}",
            fontsize=13, fontweight="bold", color="#333333", y=1.01
        )

        self._draw_lattice(
            ax=axes[0],
            G=G_before,
            pos=pos_before,
            lost_intents=lost,
            gained_intents=set(),
            title=f"Window t−1  (before drift)",
            title_color=COLOR["title_before"],
            mode="before",
        )
        self._draw_lattice(
            ax=axes[1],
            G=G_after,
            pos=pos_after,
            lost_intents=set(),
            gained_intents=gained,
            title=f"Window t  (after drift)",
            title_color=COLOR["title_after"],
            mode="after",
        )

        # Легенда
        legend_patches = [
            mpatches.Patch(color=COLOR["stable"],  label="Stable concept"),
            mpatches.Patch(color=COLOR["lost"],    label="Lost concept"),
            mpatches.Patch(color=COLOR["gained"],  label="New concept"),
            mpatches.Patch(color=COLOR["top"],     label="TOP (∅)"),
            mpatches.Patch(color=COLOR["bottom"],  label="BOTTOM (all attrs)"),
        ]
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=5,
            fontsize=8,
            framealpha=0.0,
            bbox_to_anchor=(0.5, -0.04),
        )

        # Stats strip
        n_lost   = len(lost)
        n_gained = len(gained)
        n_stable = len(stable)
        sim = len(stable) / max(len(intents_before | intents_after), 1)
        fig.text(
            0.5, -0.09,
            f"Intents: before={len(intents_before)}  after={len(intents_after)}  "
            f"stable={n_stable}  lost={n_lost}  gained={n_gained}   "
            f"Jaccard similarity={sim:.3f}",
            ha="center", fontsize=9, color="#666666",
        )

        plt.tight_layout()

        # Зберігаємо в файл якщо потрібно
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fpath = os.path.join(output_dir, f"lattice_episode_{episode_id:02d}.png")
            fig.savefig(fpath, dpi=self.dpi, bbox_inches="tight",
                        facecolor=COLOR["bg"])

        # Base64 для HTML
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi,
                    bbox_inches="tight", facecolor=COLOR["bg"])
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _draw_lattice(
        self,
        ax,
        G: nx.DiGraph,
        pos: Dict,
        lost_intents:   Set[FrozenSet],
        gained_intents: Set[FrozenSet],
        title: str,
        title_color: str,
        mode: str,  # "before" | "after"
    ):
        ax.set_facecolor(COLOR["bg"])
        ax.set_title(title, fontsize=11, color=title_color, pad=8)
        ax.axis("off")

        if len(G.nodes) == 0:
            ax.text(0.5, 0.5, "Empty lattice", ha="center", va="center",
                    transform=ax.transAxes, color="#999")
            return

        # Знаходимо max_intent_size для визначення BOTTOM вузла
        all_intent_sizes = [len(G.nodes[m]["intent"]) for m in G.nodes]
        max_intent_size = max(all_intent_sizes) if all_intent_sizes else 0

        # Кольори вузлів і їхні розміри
        node_colors = []
        node_edge_colors = []
        node_sizes = []

        for n in G.nodes:
            intent = G.nodes[n]["intent"]
            is_top    = (len(intent) == 0)
            is_bottom = (len(intent) == max_intent_size and max_intent_size > 0)

            # TOP і BOTTOM — більший розмір
            node_size = 1300 if (is_top or is_bottom) else 950
            node_sizes.append(node_size)

            # Колір: lost/gained мають пріоритет над top/bottom
            if intent in lost_intents:
                node_colors.append(COLOR["lost"])
                node_edge_colors.append("#A03030")
            elif intent in gained_intents:
                node_colors.append(COLOR["gained"])
                node_edge_colors.append("#2A7A4A")
            elif is_top:
                node_colors.append(COLOR["top"])
                node_edge_colors.append("#555555")
            elif is_bottom:
                node_colors.append(COLOR["bottom"])
                node_edge_colors.append("#333366")
            else:
                node_colors.append(COLOR["stable"])
                node_edge_colors.append("#2060A0")

        # Кольори ребер
        edge_colors = []
        for u, v in G.edges:
            iu = G.nodes[u]["intent"]
            iv = G.nodes[v]["intent"]
            if iu in lost_intents or iv in lost_intents:
                edge_colors.append(COLOR["edge_lost"])
            elif iu in gained_intents or iv in gained_intents:
                edge_colors.append(COLOR["edge_gained"])
            else:
                edge_colors.append(COLOR["edge_stable"])

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_colors,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            width=1.2,
            alpha=0.7,
            connectionstyle="arc3,rad=0.05",
        )

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            linewidths=1.5,
            edgecolors=node_edge_colors,
        )

        labels = {n: G.nodes[n]["label"] for n in G.nodes}
        nx.draw_networkx_labels(
            G, pos, labels=labels, ax=ax,
            font_size=7,
            font_color="white",
            font_weight="bold",
        )
