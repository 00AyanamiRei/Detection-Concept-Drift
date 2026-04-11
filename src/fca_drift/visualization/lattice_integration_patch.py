# """
# lattice_integration_patch.py
# ─────────────────────────────
# Припинити для існуючого HTML-звіту.
# Додає кнопку "Show Lattice" і PNG до кожного рядка таблиці епізодів.
# """

# import base64
# from typing import Optional, Set, FrozenSet
# from pathlib import Path

# # Імпорти для візуалізації
# try:
#     from .lattice_visualizer import LatticeVisualizer
#     from .lattice_snapshot_collector import LatticeSnapshotCollector
# except ImportError:
#     LatticeVisualizer = None
#     LatticeSnapshotCollector = None


# _VIZ = None

# def _init_viz():
#     global _VIZ
#     if _VIZ is None and LatticeVisualizer is not None:
#         _VIZ = LatticeVisualizer(dpi=120, figsize=(13, 4.5))
#     return _VIZ


# # ─── CSS — вставляється один раз у <head> звіту ───────────────────────────────
# LATTICE_CSS = """
# <style id="lattice-styles">
# .lc-btn {
#   background: #4A6CF7; color: #fff; border: none;
#   border-radius: 5px; padding: 4px 14px; font-size: .78rem;
#   cursor: pointer; margin-left: 8px; transition: background .15s;
#   font-weight: 500;
# }
# .lc-btn:hover { background: #3558d6; }
# .lc-btn.active { background: #c0392b; }
# .lc-graph-wrap {
#   display: none; margin-top: 10px;
#   border: 1px solid #e0e6ff; border-radius: 8px; overflow: hidden;
# }
# .lc-graph-wrap.open { display: block; }
# .lc-graph-wrap img { width: 100%; display: block; }
# </style>
# """

# # ─── JS — вставляється один раз перед </body> ─────────────────────────────────
# LATTICE_JS = """
# <script id="lattice-js">
# function toggleLattice(epId) {
#   var wrap = document.getElementById('lc-wrap-' + epId);
#   var btn  = document.getElementById('lc-btn-'  + epId);
#   if (!wrap) return;
#   wrap.classList.toggle('open');
#   if (wrap.classList.contains('open')) {
#     btn.textContent = 'Hide Lattice';
#     btn.classList.add('active');
#   } else {
#     btn.textContent = 'Show Lattice';
#     btn.classList.remove('active');
#   }
# }
# </script>
# """


# def _b64_png(
#     snapshot_collector: Optional['LatticeSnapshotCollector'],
#     ep_start: int,
#     ep_id: int,
#     dlt_max: float,
#     output_dir: Optional[str],
# ) -> Optional[str]:
#     """Повертає base64 PNG або None якщо snapshot недоступний."""
#     if snapshot_collector is None or LatticeVisualizer is None:
#         return None

#     snap = snapshot_collector.get_nearest(ep_start, window=10)
#     if snap is None:
#         return None

#     viz = _init_viz()
#     if viz is None:
#         return None

#     return viz.render_comparison_png(
#         intents_before = snap.intents_before,
#         intents_after  = snap.intents_after,
#         episode_id     = ep_id,
#         delta_lt       = dlt_max,
#         output_dir     = output_dir,
#     )


# def generate_episode_row_with_lattice(
#     episode,
#     snapshot_collector: Optional['LatticeSnapshotCollector'] = None,
#     output_dir: Optional[str] = None,
#     language: str = "en",
# ) -> str:
#     """
#     Повертає HTML для одного рядка епізоду — з кнопкою Show Lattice.

#     Параметри episode:
#       .index / .id   — номер
#       .start, .end   — часовий діапазон
#       .dlt_max       — максимальне ΔL
#       .sim_min       — мінімальна подібність
#       .dominant_type — тип дрифту
#       (опціонально) .lost_intents, .gained_intents, .stable_intents
#     """
#     ep_id    = getattr(episode, "index", getattr(episode, "id", 0))
#     ep_start = episode.start
#     ep_end   = episode.end
#     duration = ep_end - ep_start + 1
#     dlt_max  = getattr(episode, "dlt_max", 0.0)
#     sim_min  = getattr(episode, "sim_min", 0.0)
#     ep_type  = getattr(episode, "dominant_type", "unknown").upper()

#     # Текстові деталі інтенсій (якщо збережені в епізоді)
#     snap = None
#     if snapshot_collector:
#         snap = snapshot_collector.get_nearest(ep_start, window=10)

#     lost_text = gained_text = stable_text = ""
#     concepts_change = ""
#     if snap:
#         def fmt(intents):
#             items = []
#             for i in sorted(intents, key=lambda x: sorted(x)):
#                 attr_list = sorted(int(a) for a in i)
#                 items.append("{" + ", ".join(str(a) for a in attr_list) + "}")
#             return ", ".join(items) if items else "—"

#         lost_text   = fmt(snap.lost)   or "—"
#         gained_text = fmt(snap.gained) or "—"
#         stable_text = fmt(snap.stable) or "—"
#         concepts_change = f"{len(snap.intents_before)} → {len(snap.intents_after)}"

#     # PNG граф
#     b64 = _b64_png(snapshot_collector, ep_start, ep_id, dlt_max, output_dir)
#     lattice_btn = ""
#     lattice_graph = ""

#     # ✨ Головне: генерувати кнопку ЗАВЖДИ якщо є деталі
#     has_details = lost_text != "" or gained_text != "" or stable_text != ""

#     if b64:
#         # Якщо є PNG - показати його
#         lattice_btn = f'<button id="lc-btn-{ep_id}" class="lc-btn" onclick="toggleLattice({ep_id})">Show Lattice</button>'
#         lattice_graph = f'<div id="lc-wrap-{ep_id}" class="lc-graph-wrap"><img src="data:image/png;base64,{b64}" alt="Lattice comparison episode {ep_id}"/></div>'
#     elif has_details:
#         # Якщо немає PNG, але є текстові деталі - показати кнопку для розгортання деталей
#         lattice_btn = f'<button id="lc-btn-{ep_id}" class="lc-btn" style="background: #95a5a6;" onclick="toggleLattice({ep_id})">Show Details</button>'
#         lattice_graph = f'<div id="lc-wrap-{ep_id}" class="lc-graph-wrap"><div style="padding: 10px; background: #f5f5f5; color: #888; font-size: .75rem; border-top: 1px solid #ddd;"><em>Lattice graph not available. Attribute details shown above.</em></div></div>'
#     else:
#         # Якщо нема ні PNG ні деталей
#         lattice_btn = '<span style="color: #999; font-size: .8rem;">—</span>'

#     type_colors = {
#         "SUDDEN":      ("fde8e8", "c0392b"),
#         "GRADUAL":     ("e8f4fd", "2980b9"),
#         "INCREMENTAL": ("e8f8f0", "27ae60"),
#         "UNKNOWN":     ("f0f0f0", "777777"),
#     }
#     bg, fg = type_colors.get(ep_type, ("f0f0f0", "777777"))

#     return f"""
#     <tr>
#       <td><b>#{ep_id}</b></td>
#       <td>[{ep_start}, {ep_end}]</td>
#       <td>{duration}</td>
#       <td style="color:#4A6CF7;font-weight:700">{dlt_max:.4f}</td>
#       <td>{sim_min:.3f}</td>
#       <td>
#         <span style="background:#{bg};color:#{fg};border-radius:4px;
#                      padding:2px 8px;font-size:.78rem;font-weight:600">
#           {ep_type}
#         </span>
#       </td>
#       <td>{lattice_btn}</td>
#     </tr>
#     <tr>
#       <td colspan="7" style="padding:0 0 8px 0">
#         <div style="border-left:3px solid #4A6CF7;padding:10px 16px;
#                     background:#fafbff;border-radius:0 6px 6px 0">
#           <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:6px">
#             <div>
#               <span style="color:#c0392b;font-weight:600;font-size:.82rem">
#                 Lost Attributes:</span>
#               <div style="background:#fde8e8;border-radius:4px;padding:6px 8px;
#                           font-size:.78rem;margin-top:4px;font-family:monospace">
#                 {lost_text}
#               </div>
#             </div>
#             <div>
#               <span style="color:#27ae60;font-weight:600;font-size:.82rem">
#                 Gained Attributes:</span>
#               <div style="background:#e8f8f0;border-radius:4px;padding:6px 8px;
#                           font-size:.78rem;margin-top:4px;font-family:monospace">
#                 {gained_text}
#               </div>
#             </div>
#             <div>
#               <span style="color:#666;font-weight:600;font-size:.82rem">
#                 Stable Attributes:</span>
#               <div style="background:#f5f5f5;border-radius:4px;padding:6px 8px;
#                           font-size:.78rem;margin-top:4px;font-family:monospace">
#                 {stable_text}
#               </div>
#             </div>
#           </div>
#           {"<div style='font-size:.8rem;color:#888;margin-top:2px'>Concepts: " + concepts_change + "</div>" if concepts_change else ""}
#           {lattice_graph}
#         </div>
#       </td>
#     </tr>"""
"""
lattice_integration_patch.py
─────────────────────────────
Патч для існуючого HTML-звіту.
Додає кнопку "Show Lattice" і PNG до кожного рядка таблиці епізодів.

Як використовувати:
  Знайди в своєму DetailedReportGeneratorV2 місце де генерується
  HTML для рядка епізоду (там де є "Lost Attributes", "Gained Attributes").
  Замість цього виклику підстав generate_episode_row().
"""

import base64, io, os
from typing import Optional, Set, FrozenSet
from .lattice_visualizer import LatticeVisualizer
from .lattice_snapshot_collector import LatticeSnapshotCollector


_VIZ = LatticeVisualizer(dpi=120, figsize=(13, 4.5))


# ─── CSS — вставляється один раз у <head> звіту ───────────────────────────────
LATTICE_CSS = """
<style id="lattice-styles">
.lc-btn {
  background: #4A6CF7; color: #fff; border: none;
  border-radius: 5px; padding: 4px 14px; font-size: .78rem;
  cursor: pointer; margin-left: 8px; transition: background .15s;
}
.lc-btn:hover { background: #3558d6; }
.lc-btn.active { background: #c0392b; }
.lc-graph-wrap {
  display: none; margin-top: 10px;
  border: 1px solid #e0e6ff; border-radius: 8px; overflow: hidden;
}
.lc-graph-wrap.open { display: block; }
.lc-graph-wrap img { width: 100%; display: block; }
</style>
"""

# ─── JS — вставляється один раз перед </body> ─────────────────────────────────
LATTICE_JS = """
<script id="lattice-js">
function toggleLattice(epId) {
  var wrap = document.getElementById('lc-wrap-' + epId);
  var btn  = document.getElementById('lc-btn-'  + epId);
  if (!wrap) return;
  wrap.classList.toggle('open');
  if (wrap.classList.contains('open')) {
    btn.textContent = 'Hide Lattice';
    btn.classList.add('active');
  } else {
    btn.textContent = 'Show Lattice';
    btn.classList.remove('active');
  }
}
</script>
"""


def _b64_png(
    snapshot_collector: Optional[LatticeSnapshotCollector],
    ep_start: int,
    ep_id: int,
    dlt_max: float,
    output_dir: Optional[str],
) -> Optional[str]:
    """Повертає base64 PNG або None якщо snapshot недоступний."""
    if snapshot_collector is None:
        return None
    try:
        snap = snapshot_collector.get_nearest(ep_start, window=10)
        if snap is None:
            return None
        return _VIZ.render_comparison_png(
            intents_before = snap.intents_before,
            intents_after  = snap.intents_after,
            episode_id     = ep_id,
            delta_lt       = dlt_max,
            output_dir     = output_dir,
        )
    except Exception as e:
        # Fallback: return None if PNG rendering fails
        import sys
        print(f"[WARNING] PNG rendering for episode {ep_id} failed: {e}", file=sys.stderr)
        return None


def generate_episode_row(
    episode,
    snapshot_collector: Optional[LatticeSnapshotCollector] = None,
    output_dir: Optional[str] = None,
    language: str = "en",
) -> str:
    """
    Повертає HTML для одного рядка епізоду — з кнопкою Show Lattice.

    Параметри episode (підходить будь-який об'єкт з атрибутами):
      .index / .id   — номер
      .start, .end   — часовий діапазон
      .dlt_max       — максимальне ΔL
      .sim_min       — мінімальна подібність
      .dominant_type — тип дрифту
      (опціонально) .lost_intents, .gained_intents, .stable_intents
                     — вже готові множини для текстових деталей
    """
    ep_id    = getattr(episode, "index", getattr(episode, "id", 0))
    signal_start = episode.start
    signal_end = episode.end
    ep_start = getattr(episode, "start_instance_id", signal_start)
    ep_end = getattr(episode, "end_instance_id", signal_end)
    ep_center = getattr(episode, "center_instance_id", getattr(episode, "center", ep_start))
    duration = signal_end - signal_start + 1
    dlt_max  = getattr(episode, "dlt_max", 0.0)
    sim_min  = getattr(episode, "sim_min", 0.0)
    ep_type  = getattr(episode, "dominant_type", "unknown").upper()

    # Текстові деталі (якщо збережені в епізоді)
    snap = None
    if snapshot_collector:
      snap = snapshot_collector.get_nearest(ep_center, window=10)

    lost_text = gained_text = stable_text = ""
    concepts_change = ""
    if snap:
        def fmt(intents):
            return ", ".join(str(sorted(i)) for i in sorted(intents, key=lambda x: sorted(x)))
        lost_text   = fmt(snap.lost)   or "—"
        gained_text = fmt(snap.gained) or "—"
        stable_text = fmt(snap.stable) or "—"
        concepts_change = f"{len(snap.intents_before)} → {len(snap.intents_after)}"

    # PNG граф
    b64 = _b64_png(snapshot_collector, ep_center, ep_id, dlt_max, output_dir)
    lattice_btn = ""
    lattice_graph = ""
    if b64:
        lattice_btn = f"""
          <button id="lc-btn-{ep_id}" class="lc-btn"
                  onclick="toggleLattice({ep_id})">Show Lattice</button>"""
        lattice_graph = f"""
        <div id="lc-wrap-{ep_id}" class="lc-graph-wrap">
          <img src="data:image/png;base64,{b64}"
               alt="Lattice comparison episode {ep_id}"/>
        </div>"""

    type_colors = {
        "SUDDEN":      ("fde8e8", "c0392b"),
        "GRADUAL":     ("e8f4fd", "2980b9"),
        "INCREMENTAL": ("e8f8f0", "27ae60"),
        "UNKNOWN":     ("f0f0f0", "777777"),
    }
    bg, fg = type_colors.get(ep_type, ("f0f0f0", "777777"))

    return f"""
    <tr>
      <td><b>#{ep_id}</b></td>
      <td>[{ep_start}, {ep_end}]<br/><span style=\"color:#888;font-size:.75rem\">signal [{signal_start}, {signal_end}]</span></td>
      <td>{duration}</td>
      <td style="color:#4A6CF7;font-weight:700">{dlt_max:.4f}</td>
      <td>{sim_min:.3f}</td>
      <td>
        <span style="background:#{bg};color:#{fg};border-radius:4px;
                     padding:2px 8px;font-size:.78rem;font-weight:600">
          {ep_type}
        </span>
      </td>
      <td>{lattice_btn}</td>
    </tr>
    <tr>
      <td colspan="7" style="padding:0 0 8px 0">
        <div style="border-left:3px solid #4A6CF7;padding:10px 16px;
                    background:#fafbff;border-radius:0 6px 6px 0">
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:6px">
            <div>
              <span style="color:#c0392b;font-weight:600;font-size:.82rem">
                Lost Attributes:</span>
              <div style="background:#fde8e8;border-radius:4px;padding:6px 8px;
                          font-size:.78rem;margin-top:4px;font-family:monospace">
                {lost_text}
              </div>
            </div>
            <div>
              <span style="color:#27ae60;font-weight:600;font-size:.82rem">
                Gained Attributes:</span>
              <div style="background:#e8f8f0;border-radius:4px;padding:6px 8px;
                          font-size:.78rem;margin-top:4px;font-family:monospace">
                {gained_text}
              </div>
            </div>
            <div>
              <span style="color:#666;font-weight:600;font-size:.82rem">
                Stable Attributes:</span>
              <div style="background:#f5f5f5;border-radius:4px;padding:6px 8px;
                          font-size:.78rem;margin-top:4px;font-family:monospace">
                {stable_text}
              </div>
            </div>
          </div>
          {"<div style='font-size:.8rem;color:#888;margin-top:2px'>Concepts: " + concepts_change + "</div>" if concepts_change else ""}
          {lattice_graph}
        </div>
      </td>
    </tr>"""
