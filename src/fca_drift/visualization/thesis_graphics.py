"""
Thesis Graphics Exporter
Generates publication-ready graphs for diploma thesis with unified styling,
color-blind friendly palettes, and comprehensive parameter documentation.

Main Components:
- ThesisGraphicsExporter: Main class for all graph generation
- Helper functions for event merging, classification, warm-up handling
- Unified style configuration following dissertation best practices
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from collections import Counter
from scipy.stats import gaussian_kde, pearsonr


# ============================================================================
# GLOBAL STYLE CONFIGURATION (Color-Blind Safe ~ Okabe-Ito Palette)
# ============================================================================

THESIS_STYLE = {
    'colors_cb': {
        'sudden': '#E69F00',        # Orange (vermillion-like)
        'gradual': '#56B4E9',       # Sky Blue
        'incremental': '#009E73',   # Green (bluish-green)
        'recurring': '#CC79A7',     # Pink (reddish-purple)
        'unknown': '#999999'        # Gray
    },
    'colors_accents': {
        'threshold_static': '#FF0000',      # Red for static θ
        'threshold_adaptive': '#FFA500',    # Orange for μ±Nσ
        'nodrift_zone': '#00AA00',          # Green for safe zone
        'drift_highlight': '#FF1493',       # Hot pink for emphasis
        'no_drift_bg': '#E8F5E9'            # Light green fill
    },
    'fonts': {
        'family': 'DejaVu Sans',
        'size': 10,
        'axes_label': 11,
        'title': 12,
        'legend': 10
    },
    'grid': {
        'alpha': 0.3,
        'linestyle': ':',
        'linewidth': 0.7
    },
    'figure': {
        'dpi': 300,
        'tight_layout': True,
        'facecolor': 'white',
        'edgecolor': 'none'
    }
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def merge_drift_events(drift_indices: List[int],
                      drift_types: Dict[int, str],
                      merge_gap: int = 5) -> List[Tuple[int, int, str]]:
    """
    Merge nearby drift events into episodes.

    Events separated by ≤ merge_gap instances are combined.
    Dominant type of merged episode is determined by majority voting.

    Args:
        drift_indices: List of instance indices where drifts occurred
        drift_types: Dict mapping drift_index -> type string
        merge_gap: Maximum distance (in instances) to merge events

    Returns:
        List of (start_idx, end_idx, dominant_type) tuples
    """
    if not drift_indices:
        return []

    sorted_indices = sorted(drift_indices)
    merged_episodes = []
    current_episode = [sorted_indices[0]]

    for idx in sorted_indices[1:]:
        if idx - current_episode[-1] <= merge_gap:
            current_episode.append(idx)
        else:
            # Finalize current episode
            start = current_episode[0]
            end = current_episode[-1]
            types_in_episode = [drift_types.get(i, 'unknown') for i in current_episode]
            dominant = Counter(types_in_episode).most_common(1)[0][0]
            merged_episodes.append((start, end, dominant))
            current_episode = [idx]

    # Final episode
    if current_episode:
        start = current_episode[0]
        end = current_episode[-1]
        types_in_episode = [drift_types.get(i, 'unknown') for i in current_episode]
        dominant = Counter(types_in_episode).most_common(1)[0][0]
        merged_episodes.append((start, end, dominant))

    return merged_episodes


def classify_drift_severity(delta_L_history: List[float],
                           drift_indices: List[int]) -> Dict[int, str]:
    """
    Classify drifts by amplitude (z-score).

    Returns:
        Dict mapping drift_index -> severity level
            EXTREME: z > 3 (0.1% of events)
            STRONG: z ∈ [2, 3] (2%)
            MODERATE: z ∈ [1, 2] (15%)
            WEAK: z ≤ 1 (84%)
    """
    mu = np.mean(delta_L_history)
    sigma = np.std(delta_L_history)

    severity = {}
    for idx in drift_indices:
        z_score = (delta_L_history[idx] - mu) / sigma if sigma > 0 else 0

        if z_score > 3:
            severity[idx] = 'EXTREME'
        elif z_score > 2:
            severity[idx] = 'STRONG'
        elif z_score > 1:
            severity[idx] = 'MODERATE'
        else:
            severity[idx] = 'WEAK'

    return severity


def compute_rolling_threshold(delta_L_history: List[float],
                             window_size: int = 10,
                             n_sigma: float = 2.0) -> List[float]:
    """
    Compute adaptive threshold based on last N instances.

    Args:
        delta_L_history: Time series of ΔLₜ values
        window_size: Number of recent instances to consider
        n_sigma: Number of standard deviations above mean

    Returns:
        List of threshold values (one per instance)
    """
    if not delta_L_history:
        return []

    window_size = max(1, int(window_size))

    if len(delta_L_history) < window_size:
        return [np.mean(delta_L_history) + n_sigma * np.std(delta_L_history)] * len(delta_L_history)

    thresholds = []
    for i in range(len(delta_L_history)):
        start = max(0, i - window_size + 1)
        window_data = delta_L_history[start:i+1]
        threshold = np.mean(window_data) + n_sigma * np.std(window_data)
        thresholds.append(threshold)

    return thresholds


def apply_warmup(drift_indices: List[int],
                warm_up_windows: int,
                window_size: int) -> List[int]:
    """Remove drifts detected during warm-up period."""
    if warm_up_windows == 0:
        return drift_indices

    warm_up_threshold = warm_up_windows * window_size
    return [idx for idx in drift_indices if idx >= warm_up_threshold]


def moving_average(data: List[float], window: int = 10) -> List[float]:
    """Compute moving average of data."""
    if not data:
        return data

    window = max(1, int(window))
    if window == 1:
        return data

    if len(data) < window:
        return data
    return list(np.convolve(data, np.ones(window)/window, mode='valid'))


def _auto_ma_window(series_length: int, max_window: int = 15) -> int:
    """Pick a safe smoothing window that never collapses to zero."""
    if series_length <= 1:
        return 1
    return max(2, min(max_window, series_length // 10))


# ============================================================================
# THESIS GRAPHICS EXPORTER
# ============================================================================

class ThesisGraphicsExporter:
    """
    Professional graphics exporter for FCA-based concept drift detection.

    Generates publication-ready graphs with:
    - Unified styling and color-blind safe palettes
    - Comprehensive parameter documentation
    - Event merging and adaptive thresholding
    - Warm-up period handling
    - FCA-specific metrics (optional)
    """

    def __init__(self, output_dir: Path, dpi: int = 300):
        """Initialize exporter with output directory and DPI."""
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self._setup_style()

    def _setup_style(self):
        """Configure global matplotlib style."""
        style = THESIS_STYLE
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['savefig.dpi'] = self.dpi
        plt.rcParams['font.family'] = style['fonts']['family']
        plt.rcParams['font.size'] = style['fonts']['size']
        plt.rcParams['axes.labelsize'] = style['fonts']['axes_label']
        plt.rcParams['axes.titlesize'] = style['fonts']['title']
        plt.rcParams['legend.fontsize'] = style['fonts']['legend']
        plt.rcParams['grid.linestyle'] = style['grid']['linestyle']
        plt.rcParams['grid.alpha'] = style['grid']['alpha']

    def _format_title_with_params(self, base_title: str, params: Dict) -> str:
        """Format title with parameters."""
        return (f"{base_title}\n"
                f"[W={params['W']}, θ={params['θ']:.3f}, α={params['α']:.1f}, "
                f"τ_adapt={params['τ_adapt']}, warm_up={params['warm_up']}, "
                f"seed={params['seed']}]")

    def _add_params_box(self, ax, params: Dict):
        """Add parameter metadata box to axes."""
        params_text = (f"Parameters:\n"
                      f"W={params['W']}\n"
                      f"θ={params['θ']:.3f}\n"
                      f"α={params['α']:.1f}\n"
                      f"τ_adapt={params['τ_adapt']}\n"
                      f"warm_up={params['warm_up']}\n"
                      f"seed={params['seed']}")

        ax.text(0.98, 0.97, params_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8,
                        edgecolor='black', linewidth=1))

    def _add_drift_info_box(self, ax, drift_count: int, merged_count: int = None):
        """Add drift detection info box to axes (top-left corner)."""
        if merged_count is None:
            merged_count = drift_count

        drift_text = f"Detected Drifts: {drift_count}\nMerged Episodes: {merged_count}"

        ax.text(0.02, 0.98, drift_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='left',
               fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85,
                        edgecolor='darkblue', linewidth=1.5))

    def export_delta_L_analysis(self,
                               delta_L_history: List[float],
                               drift_indices: List[int],
                               drift_types: Dict[int, str],
                               theta: float,
                               merged_episodes: List[Tuple[int, int, str]],
                               warm_up_windows: int,
                               window_size: int,
                               params: Dict) -> Path:
        """
        Export advanced ΔLₜ analysis plot.

        Shows:
        - Raw and smoothed ΔLₜ curves
        - Static and adaptive thresholds
        - Merged drift episodes as regions
        - Individual drift points as markers
        - Warm-up period shaded background
        """
        fig, ax = plt.subplots(figsize=(16, 8), dpi=self.dpi)
        instances = list(range(len(delta_L_history)))

        colors = THESIS_STYLE['colors_cb']
        accents = THESIS_STYLE['colors_accents']

        # ===== WARM-UP ZONE =====
        if warm_up_windows > 0:
            warmup_end = warm_up_windows * window_size
            ax.axvspan(0, warmup_end, alpha=0.1, color='gray',
                      label='Warm-up Period', hatch='///', linewidth=0)

        # ===== RAW ΔLₜ =====
        ax.plot(instances, delta_L_history, 'b-', linewidth=0.6, alpha=0.3,
               label='Raw ΔLₜ')

        # ===== SMOOTHED ΔLₜ =====
        MA_window = _auto_ma_window(len(delta_L_history))
        if MA_window > 1 and len(delta_L_history) >= MA_window:
            smoothed = moving_average(delta_L_history, MA_window)
            ax.plot(instances[MA_window-1:], smoothed, 'darkblue', linewidth=2.5,
                   label=f'MA-smoothed (w={MA_window})', alpha=0.95, zorder=3)

        # ===== STATIC THRESHOLD =====
        ax.axhline(y=theta, linestyle='--', linewidth=2, color=accents['threshold_static'],
                  label=f'Static θ={theta:.3f}', zorder=2)

        # ===== ADAPTIVE THRESHOLD =====
        adaptive_threshold = compute_rolling_threshold(delta_L_history, window_size, n_sigma=2.0)
        ax.plot(instances, adaptive_threshold, linestyle=':', linewidth=2.5,
               color=accents['threshold_adaptive'],
               label=f'Adaptive (μ±2σ)', alpha=0.8, zorder=2)

        # ===== NO-DRIFT ZONE =====
        ax.fill_between(instances, 0, theta, alpha=0.08, color=accents['nodrift_zone'],
                       label='No-drift zone')

        # ===== MERGED EPISODES (colored regions) =====
        for start, end, dtype in merged_episodes:
            color = colors.get(dtype, colors['unknown'])
            ax.axvspan(start, end, alpha=0.2, facecolor=color, edgecolor=color,
                      linewidth=1.5, linestyle='-', zorder=1)
            # Label at midpoint
            mid = (start + end) / 2
            max_val = max(delta_L_history) if delta_L_history else 1
            ax.text(mid, max_val * 1.08, dtype.upper(), ha='center', fontsize=8,
                   fontweight='bold', color=color, zorder=4)

        # ===== INDIVIDUAL DRIFT MARKERS =====
        for drift_idx in drift_indices:
            if drift_idx < len(delta_L_history):
                dtype = drift_types.get(drift_idx, 'unknown')
                color = colors.get(dtype, colors['unknown'])
                ax.plot(drift_idx, delta_L_history[drift_idx], marker='v',
                       markersize=10, color=color, alpha=0.8, markeredgecolor='black',
                       markeredgewidth=0.5, zorder=5)

        # ===== STYLING =====
        ax.set_xlabel('Instance Number (t)', fontsize=12, fontweight='bold')
        ax.set_ylabel('ΔLₜ (Lattice Change)', fontsize=12, fontweight='bold')
        ax.set_title(self._format_title_with_params('ΔLₜ Analysis: Advanced', params),
                    fontsize=13, fontweight='bold', pad=15)

        # Legend
        legend_elements = [
            mpatches.Patch(color=colors['sudden'], alpha=0.7, label='Sudden Drift'),
            mpatches.Patch(color=colors['gradual'], alpha=0.7, label='Gradual Drift'),
            mpatches.Patch(color=colors['incremental'], alpha=0.7, label='Incremental Drift'),
            mpatches.Patch(color=colors['recurring'], alpha=0.7, label='Recurring Drift'),
            mpatches.Patch(color=accents['nodrift_zone'], alpha=0.2, label='No-drift zone'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, ncol=2,
                 frameon=True, shadow=True, fancybox=True)

        ax.grid(True, which='major', alpha=THESIS_STYLE['grid']['alpha'],
               linestyle=THESIS_STYLE['grid']['linestyle'],
               linewidth=THESIS_STYLE['grid']['linewidth'])
        ax.grid(True, which='minor', alpha=0.1, linestyle=':', linewidth=0.5)
        ax.set_xlim(0, len(delta_L_history)-1 if delta_L_history else 1)
        y_max = max(delta_L_history) if delta_L_history else 1.0
        if y_max <= 0:
            y_max = 1e-3
        ax.set_ylim(0, y_max * 1.15)

        self._add_drift_info_box(ax, len(drift_indices), len(merged_episodes))
        self._add_params_box(ax, params)
        # Explicit spacing avoids tight_layout warning with mixed subplot geometry.
        fig.subplots_adjust(top=0.94, bottom=0.06, left=0.06, right=0.98, hspace=0.38)

        output_path = self.output_dir / "01_delta_L_thesis.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"[OK] Thesis DeltaL_t plot saved to {output_path}")
        plt.close()

        return output_path

    def export_similarity_trend(self,
                               similarity_history: List[float],
                               drift_indices: List[int],
                               delta_L_history: Optional[List[float]] = None,
                               merged_episodes: Optional[List] = None,
                               params: Dict = None) -> Path:
        """
        Export Similarity trend analysis with statistical overlays.

        Shows:
        - Raw and smoothed Similarity curves
        - Mean ± 1σ band
        - Drift point markers
        - Correlation with ΔLₜ (if provided)
        """
        fig, ax = plt.subplots(figsize=(14, 6), dpi=self.dpi)
        instances = list(range(len(similarity_history)))

        colors = THESIS_STYLE['colors_cb']
        mean_sim = np.mean(similarity_history)
        std_sim = np.std(similarity_history)

        # ===== FILL MEAN ZONE =====
        ax.fill_between(instances, 0, mean_sim, alpha=0.05, color='blue')

        # ===== RAW SIMILARITY =====
        ax.plot(instances, similarity_history, 'b-', linewidth=0.6, alpha=0.35,
               label='Raw Similarity')

        # ===== SMOOTHED SIMILARITY =====
        MA_window = _auto_ma_window(len(similarity_history))
        if MA_window > 1 and len(similarity_history) >= MA_window:
            smoothed_sim = moving_average(similarity_history, MA_window)
            ax.plot(instances[MA_window-1:], smoothed_sim, 'darkblue', linewidth=2.5,
                   label=f'Smoothed (MA-{MA_window})', alpha=0.95)

        # ===== MEAN LINE =====
        ax.axhline(y=mean_sim, color='green', linestyle='--', linewidth=1.5,
                  label=f'Mean μ={mean_sim:.3f}', alpha=0.8)

        # ===== ±1σ BAND =====
        ax.fill_between(instances, mean_sim - std_sim, mean_sim + std_sim,
                       alpha=0.1, color='green', label=f'±1σ (σ={std_sim:.3f})')

        # ===== DRIFT MARKERS =====
        if drift_indices:
            drift_sims = [similarity_history[i] for i in drift_indices
                         if i < len(similarity_history)]
            valid_drift_idx = [i for i in drift_indices if i < len(similarity_history)]
            ax.scatter(valid_drift_idx, drift_sims, marker='X', s=120,
                      color='red', linewidth=2.5, edgecolors='darkred',
                      label='Detected Drift Points', zorder=5)

        # ===== STYLING =====
        ax.set_xlabel('Instance Number (t)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Lattice Similarity Score', fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.set_title('Lattice Similarity Trend (Synchronized with ΔLₜ)',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=THESIS_STYLE['grid']['alpha'],
               linestyle=THESIS_STYLE['grid']['linestyle'])

        # ===== CORRELATION ANNOTATION =====
        if delta_L_history and len(delta_L_history) == len(similarity_history):
            delta_arr = np.asarray(delta_L_history, dtype=float)
            sim_arr = np.asarray(similarity_history, dtype=float)

            if np.std(delta_arr) > 1e-12 and np.std(sim_arr) > 1e-12:
                corr, _ = pearsonr(delta_arr, sim_arr)
                corr_text = f'Correlation(ΔLₜ, Similarity) = {corr:.3f}'
            else:
                corr_text = 'Correlation(ΔLₜ, Similarity) = N/A (constant signal)'

            ax.text(0.98, 0.05, corr_text,
                   transform=ax.transAxes, fontsize=9, ha='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        if merged_episodes is None:
            merged_episodes = drift_indices
        self._add_drift_info_box(ax, len(drift_indices), len(merged_episodes))
        if params:
            self._add_params_box(ax, params)

        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.15)

        output_path = self.output_dir / "02_similarity_thesis.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"[OK] Thesis Similarity plot saved to {output_path}")
        plt.close()

        return output_path

    def export_drift_distribution(self,
                                 drift_indices: List[int],
                                 drift_types: Dict[int, str],
                                 total_instances: int,
                                 merged_episodes: List[Tuple[int, int, str]],
                                 params: Dict) -> Path:
        """
        Export comprehensive drift distribution analysis (3-panel layout).

        Panels:
        1. Pie chart of drift types with percentages
        2. Histogram + KDE of spatial distribution
        3. Stacked bar chart of types over regions + statistics table
        """
        fig = plt.figure(figsize=(16, 10), dpi=self.dpi)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.5, 0.8], hspace=0.4, wspace=0.3)

        colors = THESIS_STYLE['colors_cb']

        # ===== PANEL 1: PIE CHART =====
        ax_pie = fig.add_subplot(gs[0, 0])

        # Use merged episodes types for consistency with displayed drift count (SSOT)
        # merged_episodes are tuples: (start, end, type) or DriftEpisode objects
        if merged_episodes:
            merged_types = []
            for ep in merged_episodes:
                if isinstance(ep, tuple):
                    merged_types.append(ep[2])  # 3rd element is type
                else:
                    merged_types.append(ep.dominant_type)  # DriftEpisode object
            type_counts = Counter(merged_types)
        else:
            type_counts = Counter(drift_types.values())

        if not type_counts:
            type_counts = {'No Drifts': 1}

        types_list = list(type_counts.keys())
        counts_list = list(type_counts.values())
        colors_pie = [colors.get(t, colors['unknown']) for t in types_list]

        wedges, texts, autotexts = ax_pie.pie(
            counts_list, labels=types_list, colors=colors_pie,
            autopct=lambda pct: f'{int(pct*sum(counts_list)/100)}\n({pct:.1f}%)',
            startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'},
            explode=[0.08] * len(types_list), shadow=True, pctdistance=0.85
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)

        ax_pie.set_title(f'Drift Type Distribution\n(Total: {len(merged_episodes)} episodes)',
                        fontsize=11, fontweight='bold', pad=10)

        # Add drift info box on pie chart
        drift_info_text = f"Detected: {len(drift_indices)}\nMerged: {len(merged_episodes)}"
        ax_pie.text(0.5, -0.05, drift_info_text, transform=ax_pie.transAxes,
                   fontsize=10, fontweight='bold', ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85,
                            edgecolor='darkblue', linewidth=1.5))

        # ===== PANEL 2: HISTOGRAM + KDE =====
        ax_hist = fig.add_subplot(gs[0, 1])

        if drift_indices:
            n_bins = min(30, len(drift_indices)//2 + 1)
            counts, bins, patches = ax_hist.hist(drift_indices, bins=n_bins,
                                                color='#3498db', alpha=0.6,
                                                edgecolor='black', linewidth=0.8,
                                                label='Drift Count')

            # KDE overlay
            if len(drift_indices) > 3:
                try:
                    kde = gaussian_kde(drift_indices, bw_method=0.2)
                    x_range = np.linspace(0, total_instances, 300)
                    kde_vals = kde(x_range)
                    max_count = counts.max() if len(counts) > 0 else 1
                    kde_normalized = kde_vals * (max_count / kde_vals.max())
                    ax_hist.plot(x_range, kde_normalized, 'r-', linewidth=3,
                               label='Density (KDE)', alpha=0.85, zorder=3)
                except Exception as e:
                    print(f"[WARN] KDE computation failed: {e}")

            if len(drift_indices) > 0:
                ax_hist.legend(fontsize=10, loc='upper right')
        else:
            ax_hist.text(0.5, 0.5, 'No Drifts', ha='center', va='center',
                        transform=ax_hist.transAxes, fontsize=12)

        # ===== PANEL 3: STACKED BAR CHART =====
        ax_stacked = fig.add_subplot(gs[1, :])

        n_regions = 5
        region_size = total_instances // n_regions
        region_data = {dtype: [] for dtype in types_list}

        for region_idx in range(n_regions):
            start = region_idx * region_size
            end = (region_idx + 1) * region_size if region_idx < n_regions - 1 else total_instances
            region_indices = [i for i in drift_indices if start <= i < end]

            for dtype in types_list:
                count = sum(1 for i in region_indices if drift_types.get(i) == dtype)
                region_data[dtype].append(count)

        x_pos = np.arange(n_regions)
        width = 0.6
        bottom = np.zeros(n_regions)

        for dtype in types_list:
            counts_region = region_data[dtype]
            ax_stacked.bar(x_pos, counts_region, width, label=dtype,
                          bottom=bottom, color=colors.get(dtype, colors['unknown']),
                          edgecolor='black', linewidth=0.8)
            bottom += np.array(counts_region)

        ax_stacked.set_xlabel('Stream Region', fontsize=11, fontweight='bold')
        ax_stacked.set_ylabel('Drift Count per Region', fontsize=11, fontweight='bold')
        ax_stacked.set_title('Temporal Distribution of Drift Types',
                           fontsize=11, fontweight='bold', pad=10)
        ax_stacked.set_xticks(x_pos)
        ax_stacked.set_xticklabels([f'R{i+1}' for i in range(n_regions)])
        ax_stacked.legend(fontsize=10, loc='upper left', ncol=len(types_list))
        ax_stacked.grid(True, axis='y', alpha=THESIS_STYLE['grid']['alpha'],
                       linestyle=THESIS_STYLE['grid']['linestyle'])

        # ===== PANEL 4: STATISTICS BOX =====
        ax_stats = fig.add_subplot(gs[2, :])
        ax_stats.axis('off')

        # SSOT: Use merged episodes for "Total Drifts" (consistent with HTML)
        # raw_count is available for reference but not used in main statistics
        total_drifts = len(merged_episodes)  # Use MERGED for consistency with HTML
        drift_rate = (total_drifts / total_instances * 100) if total_instances > 0 else 0

        # Calculate avg distance only from merged episode starts
        merged_starts = [ep[0] if isinstance(ep, tuple) else ep.start for ep in merged_episodes]
        avg_distance = np.mean(np.diff(merged_starts)) if len(merged_starts) > 1 else 0

        # Type breakdown from type_counts (which should be from merged episodes)
        type_breakdown = '  |  '.join([f"{t}: {c} ({100*c/max(total_drifts,1):.1f}%)"
                                      for t, c in type_counts.items()]) if total_drifts > 0 else "No drifts"

        stats_text = (
            f"DRIFT DETECTION STATISTICS\n"
            f"{'─'*85}\n"
            f"Total Drifts: {total_drifts:>4}  |  Drift Rate: {drift_rate:>6.2f}%  |  "
            f"Total Instances: {total_instances}\n"
            f"Avg Distance: {avg_distance:>6.1f}  |  Merged Episodes: {len(merged_episodes):>2}  |  "
            f"Raw Detections: {len(drift_indices):>2}  |  Merge Efficiency: {len(merged_episodes)/max(len(drift_indices),1)*100:>5.1f}%\n"
            f"{'─'*85}\n"
            f"Type Breakdown: {type_breakdown}\n"
            f"{'─'*85}\n"
            f"Params: W={params['W']}  θ={params['θ']:.3f}  α={params['α']:.1f}  "
            f"τ_adapt={params['τ_adapt']}  warm_up={params['warm_up']}"
        )

        ax_stats.text(0.05, 0.5, stats_text, fontsize=9, fontfamily='monospace',
                     verticalalignment='center', transform=ax_stats.transAxes,
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8,
                              edgecolor='black', linewidth=1.5))

        # Explicit spacing avoids tight_layout warning with GridSpec geometry
        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.12, hspace=0.4)

        output_path = self.output_dir / "03_drift_distribution_thesis.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"[OK] Thesis Drift Distribution plot saved to {output_path}")
        plt.close()

        return output_path

    def export_synchronized_view(self,
                                delta_L_history: List[float],
                                similarity_history: List[float],
                                drift_indices: List[int],
                                drift_types: Dict[int, str],
                                theta: float,
                                merged_episodes: Optional[List] = None,
                                params: Dict = None) -> Path:
        """
        Export synchronized 2-panel view: Similarity (top) + ΔLₜ (bottom).

        Shares X-axis for direct comparison.
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), dpi=self.dpi, sharex=True)
        instances = list(range(len(similarity_history)))

        colors = THESIS_STYLE['colors_cb']
        accents = THESIS_STYLE['colors_accents']

        # ===== TOP: SIMILARITY =====
        mean_sim = np.mean(similarity_history)
        std_sim = np.std(similarity_history)

        axes[0].plot(instances, similarity_history, 'b-', linewidth=0.6, alpha=0.3,
                    label='Raw Similarity')

        MA_window = _auto_ma_window(len(similarity_history))
        if MA_window > 1 and len(similarity_history) >= MA_window:
            smoothed_sim = moving_average(similarity_history, MA_window)
            axes[0].plot(instances[MA_window-1:], smoothed_sim, 'darkblue', linewidth=2.5,
                        label=f'Smoothed (MA-{MA_window})', alpha=0.95)

        axes[0].axhline(y=mean_sim, color='green', linestyle='--', linewidth=1.5,
                       label=f'Mean={mean_sim:.3f}')
        axes[0].fill_between(instances, mean_sim - std_sim, mean_sim + std_sim,
                            alpha=0.1, color='green')

        if drift_indices:
            drift_sims = [similarity_history[i] for i in drift_indices if i < len(similarity_history)]
            valid_idx = [i for i in drift_indices if i < len(similarity_history)]
            axes[0].scatter(valid_idx, drift_sims, marker='X', s=100, color='red',
                           linewidth=2, edgecolors='darkred', zorder=5)

        axes[0].set_ylabel('Similarity Score', fontsize=11, fontweight='bold')
        axes[0].set_ylim([0, 1.05])
        axes[0].set_title('Synchronized Drift Indicators: Similarity and ΔLₜ',
                         fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper right', fontsize=10)
        axes[0].grid(True, alpha=THESIS_STYLE['grid']['alpha'],
                    linestyle=THESIS_STYLE['grid']['linestyle'])

        # Add drift info on top panel
        if merged_episodes is None:
            merged_episodes = drift_indices
        drift_text = f"Detected: {len(drift_indices)}\nMerged: {len(merged_episodes)}"
        axes[0].text(0.02, 0.98, drift_text, transform=axes[0].transAxes,
                    fontsize=10, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85,
                             edgecolor='darkblue', linewidth=1.5))

        # ===== BOTTOM: ΔLₜ =====
        axes[1].plot(instances, delta_L_history, 'r-', linewidth=0.6, alpha=0.3,
                    label='Raw ΔLₜ')

        if MA_window > 1 and len(delta_L_history) >= MA_window:
            smoothed_delta = moving_average(delta_L_history, MA_window)
            axes[1].plot(instances[MA_window-1:], smoothed_delta, 'darkred', linewidth=2.5,
                        label=f'Smoothed (MA-{MA_window})', alpha=0.95)

        adaptive_th = compute_rolling_threshold(delta_L_history, window_size=10, n_sigma=2.0)
        axes[1].plot(instances, adaptive_th, linestyle=':', linewidth=2.5,
                    color=accents['threshold_adaptive'],
                    label='Adaptive (μ±2σ)', alpha=0.8, zorder=2)

        axes[1].axhline(y=theta, linestyle='--', linewidth=2, color=accents['threshold_static'],
                       label=f'Static θ={theta:.3f}', zorder=2)

        axes[1].fill_between(instances, 0, theta, alpha=0.08, color=accents['nodrift_zone'])

        if drift_indices:
            drift_delta = [delta_L_history[i] for i in drift_indices if i < len(delta_L_history)]
            valid_idx = [i for i in drift_indices if i < len(delta_L_history)]
            axes[1].scatter(valid_idx, drift_delta, marker='X', s=100, color='red',
                           linewidth=2, edgecolors='darkred', zorder=5)

        axes[1].set_xlabel('Instance Number (t)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('ΔLₜ Value', fontsize=11, fontweight='bold')
        axes[1].legend(loc='upper right', fontsize=10)
        axes[1].grid(True, alpha=THESIS_STYLE['grid']['alpha'],
                    linestyle=THESIS_STYLE['grid']['linestyle'])

        # Explicit spacing avoids tight_layout warning with sharex subplot geometry
        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.12, hspace=0.3)

        output_path = self.output_dir / "04_synchronized_view_thesis.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"[OK] Thesis Synchronized View plot saved to {output_path}")
        plt.close()

        return output_path

    def export_all(self,
                  delta_L_history: List[float],
                  similarity_history: List[float],
                  drift_indices: List[int],
                  drift_types: Dict[int, str],
                  window_size: int,
                  theta: float,
                  alpha: float,
                  adaptive_window: int,
                  warm_up_windows: int,
                  merge_gap: int,
                  seed: int,
                  merged_episodes: Optional[List[Tuple[int, int, str]]] = None) -> Dict[str, Path]:
        """
        Export all thesis graphics in one call.

        Args:
            merged_episodes: Pre-computed merged episodes (optional). If None, will be computed.

        Returns:
            Dict mapping graph names to output paths
        """
        # Prepare parameters dict
        params = {
            'W': window_size,
            'θ': theta,
            'α': alpha,
            'τ_adapt': adaptive_window,
            'warm_up': warm_up_windows,
            'merge_gap': merge_gap,
            'seed': seed
        }

        # Apply warm-up filtering
        drift_indices_filtered = apply_warmup(drift_indices, warm_up_windows, window_size)
        drift_types_filtered = {k: v for k, v in drift_types.items() if k in drift_indices_filtered}

        # Merge events (use provided or compute)
        if merged_episodes is None:
            merged_episodes = merge_drift_events(drift_indices_filtered, drift_types_filtered, merge_gap)

        print("\n[THESIS GRAPHICS EXPORT] Starting...")

        # Generate all plots
        results = {
            'delta_L_analysis': self.export_delta_L_analysis(
                delta_L_history, drift_indices_filtered, drift_types_filtered,
                theta, merged_episodes, warm_up_windows, window_size, params
            ),
            'similarity_trend': self.export_similarity_trend(
                similarity_history, drift_indices_filtered, delta_L_history, merged_episodes, params
            ),
            'drift_distribution': self.export_drift_distribution(
                drift_indices_filtered, drift_types_filtered, len(similarity_history),
                merged_episodes, params
            ),
            'synchronized_view': self.export_synchronized_view(
                delta_L_history, similarity_history, drift_indices_filtered,
                drift_types_filtered, theta, merged_episodes, params
            )
        }

        print(f"[OK] All thesis graphics exported to {self.output_dir}\n")

        return results
