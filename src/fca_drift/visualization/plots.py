"""
Visualization - Detailed analysis plots with drift detection
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime


# Color mapping for drift types
colors_map = {
    'sudden': '#E74C3C',       # Red
    'gradual': '#F39C12',      # Orange
    'incremental': '#F1C40F',  # Gold
    'unknown': '#95A5A6'       # Gray
}


def plot_delta_L(delta_L_history: List[float],
                 drift_indices: List[int],
                 threshold: float,
                 output_path: Optional[Path] = None,
                 title: str = "ΔL_t over time",
                 drift_types: Optional[Dict[int, str]] = None):
    """
    Improved plot of ΔL_t time series with:
    - Line plot instead of scatter
    - Moving average for trend visualization
    - Adaptive threshold visualization
    - Drift regions as vertical spans
    - Fill zones for no-drift area
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    instances = list(range(len(delta_L_history)))

    if drift_types is None:
        drift_types = {idx: 'unknown' for idx in drift_indices}

    # 1. Plot raw ΔL_t as line (light)
    ax.plot(instances, delta_L_history, 'b-', linewidth=0.8, alpha=0.4, label='Raw ΔL_t')

    # 2. Add moving average for trend (dark line)
    window_size = 10
    if len(delta_L_history) >= window_size:
        smoothed = np.convolve(delta_L_history, np.ones(window_size)/window_size, mode='valid')
        smoothed_instances = instances[window_size-1:]
        ax.plot(smoothed_instances, smoothed, 'darkblue', linewidth=2.5,
                label=f'Moving Average (window={window_size})', alpha=0.9, zorder=3)

    # 3. Static threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
               label=f'Static Threshold θ={threshold:.3f}', zorder=2)

    # 4. Adaptive threshold (μ + 2σ)
    if len(delta_L_history) >= 10:
        mu = np.mean(delta_L_history)
        sigma = np.std(delta_L_history)
        adaptive_th = mu + 2.0 * sigma
        ax.axhline(y=adaptive_th, color='orange', linestyle=':', linewidth=2,
                   label=f'Adaptive Threshold (μ+2σ={adaptive_th:.3f})', zorder=2)

    # 5. Fill no-drift zone (green area)
    ax.fill_between(instances, 0, threshold, alpha=0.08, color='green', label='No-drift zone')

    # 6. Drift regions as vertical spans with colors
    drift_color_map = {
        'sudden': '#FF6B6B',
        'gradual': '#FFA500',
        'incremental': '#FFD700',
        'unknown': '#CCCCCC'
    }

    for drift_idx in drift_indices:
        if drift_idx < len(delta_L_history):
            drift_type = drift_types.get(drift_idx, 'unknown')
            color = drift_color_map.get(drift_type, '#CCCCCC')

            # Vertical span (±5 instances)
            ax.axvspan(max(0, drift_idx-5), min(len(delta_L_history)-1, drift_idx+5),
                       alpha=0.15, color=color, zorder=1)

            # Marker at top
            if drift_idx < len(delta_L_history):
                ax.plot(drift_idx, max(delta_L_history) * 1.08, 'v',
                       color=color, markersize=10, alpha=0.8, zorder=4, markeredgecolor='black', markeredgewidth=0.5)

    # Legend
    legend_elements = [
        mpatches.Patch(color=drift_color_map['sudden'], label='Sudden Drift', alpha=0.7),
        mpatches.Patch(color=drift_color_map['gradual'], label='Gradual Drift', alpha=0.7),
        mpatches.Patch(color=drift_color_map['incremental'], label='Incremental Drift', alpha=0.7),
        mpatches.Patch(color='green', alpha=0.2, label='No-drift zone'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, ncol=2)

    ax.set_xlabel('Instance Number', fontsize=11)
    ax.set_ylabel('ΔL_t (Lattice Change)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_ylim(0, max(delta_L_history) * 1.15 if delta_L_history else 1)
    ax.set_xlim(0, len(delta_L_history)-1 if delta_L_history else 1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved detailed plot to {output_path}")

    plt.close()


def plot_drift_distribution(drift_indices: List[int],
                            total_instances: int,
                            drift_types: Optional[Dict[int, str]] = None,
                            output_path: Optional[Path] = None):
    """
    Improved drift distribution plot with:
    - Pie chart for type distribution (with percentages)
    - Histogram + KDE for spatial distribution
    - Better statistics display
    """
    if drift_types is None:
        drift_types = {idx: 'unknown' for idx in drift_indices}

    # Count drift types
    type_counts = {}
    for idx in drift_indices:
        dtype = drift_types.get(idx, 'unknown')
        type_counts[dtype] = type_counts.get(dtype, 0) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # LEFT PANEL: Pie chart of drift types
    if type_counts:  # Only if there are drifts
        types_list = list(type_counts.keys())
        counts_list = list(type_counts.values())
        colors_pie = [colors_map.get(t, '#95A5A6') for t in types_list]

        wedges, texts, autotexts = axes[0].pie(
            counts_list, labels=types_list, colors=colors_pie,
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 10, 'weight': 'bold'},
            explode=[0.05] * len(types_list)  # Slightly separate all slices
        )

        # Enhance autotext
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)

        axes[0].set_title(f'Drift Types Distribution\n(Total: {len(drift_indices)} drifts)',
                         fontsize=12, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'No Drifts Detected', ha='center', va='center',
                    fontsize=14, fontweight='bold', transform=axes[0].transAxes)
        axes[0].set_title('Drift Types Distribution', fontsize=12, fontweight='bold')

    # RIGHT PANEL: Histogram + KDE of spatial distribution
    if drift_indices:
        axes[1].hist(drift_indices, bins=min(30, len(drift_indices)//2 + 1),
                    color='#FF7675', alpha=0.6, edgecolor='black', linewidth=0.8)

        # Add KDE (kernel density estimation)
        try:
            from scipy.stats import gaussian_kde
            if len(drift_indices) > 3:  # Need at least 4 points for KDE
                kde = gaussian_kde(drift_indices, bw_method=0.2)
                x_range = np.linspace(0, total_instances, 300)
                kde_values = kde(x_range)

                # Normalize KDE to match histogram scale
                max_hist = axes[1].get_ylim()[1]
                kde_normalized = kde_values * (max_hist / kde_values.max())

                axes[1].plot(x_range, kde_normalized, 'r-', linewidth=3,
                            label='Drift Density (KDE)', alpha=0.8, zorder=3)
                axes[1].legend(fontsize=10)
        except ImportError:
            pass

    axes[1].set_xlabel('Instance Number', fontsize=11)
    axes[1].set_ylabel('Drift Count', fontsize=11)

    drift_rate = (len(drift_indices)/total_instances*100) if total_instances > 0 else 0.0
    axes[1].set_title(f'Spatial Distribution of Drifts\n'
                     f'(Drift Rate: {drift_rate:.2f}%)',
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y', linestyle=':')

    # Statistics box
    stats_text = f"Total Drifts: {len(drift_indices)}\nTotal Instances: {total_instances}\nDrift Rate: {drift_rate:.2f}%"
    axes[1].text(0.98, 0.97, stats_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='black', linewidth=1))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved distribution plot to {output_path}")

    plt.close()


def plot_similarity_vs_delta_L(similarity_history: List[float],
                               delta_L_history: List[float],
                               drift_indices: List[int],
                               output_path: Optional[Path] = None):
    """
    Improved synchronized plot of Similarity and ΔL_t with:
    - Smoothing for both curves
    - Multiple threshold lines
    - Better drift point visualization
    - Reference lines for mean values
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    instances = list(range(len(similarity_history)))

    # ===== TOP PANEL: Similarity Score =====
    # Raw similarity (light)
    axes[0].plot(instances, similarity_history, 'b-', linewidth=0.8, alpha=0.3, label='Raw Similarity')

    # Smoothed similarity (dark)
    window_sim = 15
    if len(similarity_history) >= window_sim:
        smoothed_sim = np.convolve(similarity_history, np.ones(window_sim)/window_sim, mode='valid')
        axes[0].plot(instances[window_sim-1:], smoothed_sim, 'darkblue', linewidth=2.5,
                    label=f'Smoothed (MA-{window_sim})', alpha=0.9)

    # Mean line
    mean_sim = np.mean(similarity_history)
    axes[0].axhline(y=mean_sim, color='green', linestyle='--', linewidth=1.5,
                   label=f'Mean = {mean_sim:.3f}', alpha=0.7)

    # Drift points
    drift_sims = [similarity_history[i] for i in drift_indices if i < len(similarity_history)]
    drift_inds = [i for i in drift_indices if i < len(similarity_history)]
    if drift_inds:
        axes[0].scatter(drift_inds, drift_sims, color='red', s=120, marker='X',
                       linewidth=2.5, label='Drift Points', zorder=5, edgecolors='darkred')

    axes[0].set_ylabel('Similarity Score', fontsize=11, fontweight='bold')
    axes[0].set_title('Lattice Similarity Over Time (Synchronized View)', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3, linestyle=':')
    axes[0].set_ylim([0, 1.05])
    axes[0].fill_between(instances, 0, 1, alpha=0.02, color='blue')

    # ===== BOTTOM PANEL: ΔL_t =====
    # Raw ΔL_t (light)
    axes[1].plot(instances, delta_L_history, 'r-', linewidth=0.8, alpha=0.3, label='Raw ΔL_t')

    # Smoothed ΔL_t (dark)
    window_delta = 15
    if len(delta_L_history) >= window_delta:
        smoothed_delta = np.convolve(delta_L_history, np.ones(window_delta)/window_delta, mode='valid')
        axes[1].plot(instances[window_delta-1:], smoothed_delta, 'darkred', linewidth=2.5,
                    label=f'Smoothed (MA-{window_delta})', alpha=0.9)

    # Static threshold
    threshold_static = np.mean(delta_L_history) + 2 * np.std(delta_L_history) if len(delta_L_history) > 1 else 0.1
    axes[1].axhline(y=threshold_static, color='red', linestyle='--', linewidth=2,
                   label=f'Adaptive Threshold (μ+2σ) = {threshold_static:.3f}', zorder=2)

    # Fill no-drift zone
    axes[1].fill_between(instances, 0, threshold_static, alpha=0.08, color='green', label='No-drift zone')

    # Drift points
    drift_delta = [delta_L_history[i] for i in drift_indices if i < len(delta_L_history)]
    drift_inds2 = [i for i in drift_indices if i < len(delta_L_history)]
    if drift_inds2:
        axes[1].scatter(drift_inds2, drift_delta, color='red', s=120, marker='X',
                       linewidth=2.5, label='Drift Points', zorder=5, edgecolors='darkred')

    axes[1].set_xlabel('Instance Number', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('ΔL_t Value', fontsize=11, fontweight='bold')
    axes[1].set_title('Lattice Change (ΔL_t) Over Time', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved similarity vs delta plot to {output_path}")

    plt.close()


def create_lattice_animator(lattice_history: List[Any],
                           drift_indices: List[int],
                           output_dir: Optional[Path] = None,
                           save_frames: bool = False):
    """
    Create interactive lattice animation using matplotlib slider.

    Args:
        lattice_history: List of ConceptLattice objects over time
        drift_indices: List of instance indices where drifts occurred
        output_dir: Optional directory to save frame images
        save_frames: Whether to save individual frames as PNG images

    Returns:
        Dictionary with visualization info
    """
    if not lattice_history:
        print("[WARN] No lattice history provided for animation")
        return {}

    drift_set = set(drift_indices)

    print(f"[INFO] Creating lattice animator for {len(lattice_history)} frames")

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))

    # Main lattice plot (80% height)
    ax_lattice = plt.subplot2grid((10, 2), (0, 0), rowspan=8, colspan=2)

    # Stats area (15% height)
    ax_stats = plt.subplot2grid((10, 2), (8, 0), rowspan=1, colspan=2)

    # Slider area (5% height)
    ax_slider = plt.subplot2grid((10, 2), (9, 0), rowspan=1, colspan=2)

    # Current state container
    current_step = {'value': 0}

    def draw_lattice(step_idx):
        """Draw lattice at given step"""
        if step_idx < 0 or step_idx >= len(lattice_history):
            return

        lattice = lattice_history[step_idx]
        ax_lattice.clear()

        # Draw simple hierarchical lattice
        if hasattr(lattice, 'concepts') and lattice.concepts:
            # Create positions based on levels
            pos = {}
            max_level = len(lattice.levels) if hasattr(lattice, 'levels') else 1

            for level_idx, level_concepts in enumerate(lattice.levels if hasattr(lattice, 'levels') else [lattice.concepts]):
                n_concepts = len(level_concepts)
                y = max_level - level_idx - 1

                for i, concept in enumerate(level_concepts):
                    try:
                        concept_id = lattice.concepts.index(concept)
                        x = (i + 1) / (n_concepts + 1)
                        pos[concept_id] = (x, y)
                    except (ValueError, AttributeError):
                        pass

            # Draw edges first (hierarchy)
            if hasattr(lattice, 'hierarchy'):
                for parent_id, children_ids in lattice.hierarchy.items():
                    if parent_id not in pos:
                        continue
                    x1, y1 = pos[parent_id]

                    for child_id in children_ids:
                        if child_id not in pos:
                            continue
                        x2, y2 = pos[child_id]
                        ax_lattice.plot([x1, x2], [y1, y2], 'gray', linewidth=1.5, alpha=0.6, zorder=1)

            # Draw nodes (concepts)
            for concept_id, (x, y) in pos.items():
                concept = lattice.concepts[concept_id]

                # Node color based on type
                if hasattr(lattice, 'top_concept') and concept == lattice.top_concept:
                    color = '#90EE90'  # Light green
                    label_text = 'TOP'
                elif hasattr(lattice, 'bottom_concept') and concept == lattice.bottom_concept:
                    color = '#FFB6C6'  # Light red
                    label_text = 'BOT'
                else:
                    color = '#87CEEB'  # Sky blue
                    label_text = f'{concept_id}'

                # Draw node
                circle = plt.Circle((x, y), 0.03, color=color, ec='black', linewidth=2, zorder=3)
                ax_lattice.add_patch(circle)

                # Label
                ax_lattice.text(x, y, label_text, ha='center', va='center',
                              fontsize=7, fontweight='bold', zorder=4)

                # Info text
                extent_size = len(concept.extent) if hasattr(concept, 'extent') else 0
                intent_size = len(concept.intent) if hasattr(concept, 'intent') else 0
                ax_lattice.text(x, y-0.08, f'{extent_size}|{intent_size}',
                              ha='center', va='top', fontsize=5, alpha=0.7, zorder=2)

        ax_lattice.set_xlim(-0.1, 1.1)
        ax_lattice.set_ylim(-0.5, max_level if max_level > 0 else 1)
        ax_lattice.axis('off')

        # Title with drift indicator
        drift_indicator = " 🔴 DRIFT!" if step_idx in drift_set else ""
        ax_lattice.set_title(
            f'Concept Lattice - Step {step_idx}{drift_indicator}',
            fontsize=12, fontweight='bold',
            color='red' if drift_indicator else 'black',
            pad=10
        )

    def draw_stats(step_idx):
        """Draw statistics panel"""
        ax_stats.clear()
        ax_stats.axis('off')

        if step_idx < len(lattice_history):
            lattice = lattice_history[step_idx]

            n_concepts = len(lattice.concepts) if hasattr(lattice, 'concepts') else 0
            n_levels = len(lattice.levels) if hasattr(lattice, 'levels') else 0
            is_drift = "✓ DRIFT" if step_idx in drift_set else "✗ No drift"

            stats_text = (
                f"Concepts: {n_concepts}  |  "
                f"Levels: {n_levels}  |  "
                f"Status: {is_drift}"
            )

            ax_stats.text(0.5, 0.5, stats_text,
                         ha='center', va='center',
                         fontsize=11, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                                  edgecolor='black', linewidth=1.5))

    def update_slider(val):
        """Slider callback"""
        step = int(slider.val)
        current_step['value'] = step
        draw_lattice(step)
        draw_stats(step)
        fig.canvas.draw_idle()

    # Create slider
    slider = plt.Slider(
        ax_slider, 'Step', 0, len(lattice_history)-1,
        valinit=0, valstep=1, color='#1f77b4', alpha=0.7
    )
    slider.on_changed(update_slider)

    # Initial render
    draw_lattice(0)
    draw_stats(0)

    plt.tight_layout()

    # Save frames if requested
    if save_frames and output_dir:
        try:
            frames_dir = Path(output_dir) / 'lattice_frames'
            frames_dir.mkdir(parents=True, exist_ok=True)

            print(f"[INFO] Saving lattice frames to {frames_dir}...")
            for i in range(len(lattice_history)):
                slider.set_val(i)
                fig.savefig(frames_dir / f'lattice_frame_{i:04d}.png',
                           dpi=150, bbox_inches='tight')
                if (i + 1) % max(1, len(lattice_history)//5) == 0:
                    print(f"  [PROGRESS] Saved {i+1}/{len(lattice_history)} frames")

            print(f"[OK] Saved {len(lattice_history)} lattice frames")
        except Exception as e:
            print(f"[ERROR] Could not save frames: {e}")

    return {
        'total_frames': len(lattice_history),
        'total_drifts': len(drift_indices),
        'successfully_created': True
    }


def create_analysis_summary(delta_L_history: List[float],
                           similarity_history: List[float],
                           drift_indices: List[int],
                           drift_types: Dict[int, str],
                           output_dir: Path):
    """
    Create a comprehensive visual summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Main ΔL with drift markers
    plot_delta_L(delta_L_history, drift_indices,
                np.mean(delta_L_history) + 2*np.std(delta_L_history),
                output_path=output_dir / "01_delta_L_analysis.png",
                drift_types=drift_types)

    # Plot 2: Drift distribution
    plot_drift_distribution(drift_indices, len(delta_L_history), drift_types,
                          output_path=output_dir / "02_drift_distribution.png")

    # Plot 3: Similarity vs ΔL
    plot_similarity_vs_delta_L(similarity_history, delta_L_history, drift_indices,
                              output_path=output_dir / "03_similarity_vs_delta.png")

    print(f"\n[OK] All analysis plots saved to {output_dir}")


# Color mapping for drift types
colors_map = {
    'sudden': '#FF0000',     # Red
    'gradual': '#FFA500',    # Orange
    'incremental': '#FFD700', # Gold
    'unknown': '#FF1493'     # DeepPink
}


if __name__ == "__main__":
    # Test with example data
    delta_L = np.concatenate([
        np.random.normal(0.2, 0.1, 50),  # Normal period
        np.random.normal(0.7, 0.1, 10),  # Drift period
        np.random.normal(0.2, 0.1, 40),  # Normal period again
    ])

    drifts = [50, 51, 52, 55, 60]
    drift_types = {idx: 'gradual' for idx in drifts}

    plot_delta_L(delta_L, drifts, 0.4, drift_types=drift_types,
                title="Test: ΔL with Drift Detection")
