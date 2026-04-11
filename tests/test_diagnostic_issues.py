"""
Diagnostic Test Suite - Identify all Phase 6 Issues
Tests specific problems reported by user:
1. Episode aggregation not working (always duration 1, same type)
2. KPI values garbage/malformed
3. Graph labels/legends missing
4. Dev content leaking into report
5. PDF has local paths
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fca_drift.utils.drift_aggregator_v2 import DriftAggregatorV2
import numpy as np


def test_episode_duration_and_merging():
    """Test if episodes are properly merged (not all duration=1)"""
    print("\n" + "="*80)
    print("TEST 1: Episode Duration & Merging")
    print("="*80)

    # Create synthetic data with close drifts that should merge
    delta_L = np.array([0.1]*100, dtype=float)
    # Create 3 clusters of drifts:
    delta_L[30:33] = [0.8, 0.85, 0.75]  # Cluster 1 (indices 30-32)
    delta_L[50:53] = [0.7, 0.75, 0.68]  # Cluster 2 (indices 50-52)
    delta_L[70:75] = [0.6, 0.65, 0.62, 0.64, 0.59]  # Cluster 3 (indices 70-74)

    similarity = np.ones(100) * 0.5
    similarity[30:33] = [0.2, 0.15, 0.25]
    similarity[50:53] = [0.3, 0.25, 0.32]
    similarity[70:75] = [0.4, 0.35, 0.38, 0.36, 0.41]

    drift_indices = [30, 31, 32, 50, 51, 52, 70, 71, 72, 73, 74]

    agg = DriftAggregatorV2(
        delta_L_history=delta_L,
        similarity_history=similarity,
        drift_indices=drift_indices,
        window_size=50,
        warm_up_windows=2,
        merge_gap=5,
        cooldown=10
    )

    episodes = agg.get_merged_episodes()

    print(f"Raw drifts: {len(drift_indices)}")
    print(f"Merged episodes: {len(episodes)}")

    all_duration_1 = all(ep.episode_length() == 1 for ep in episodes)
    print(f"All episodes have duration 1: {all_duration_1}")

    if episodes:
        print(f"\nEpisode details:")
        for i, ep in enumerate(episodes):
            print(f"  Ep {i+1}: start={ep.start}, end={ep.end}, duration={ep.episode_length()}, type={ep.dominant_type}")

    # Check if merger worked
    expected_episodes = 3  # Should have 3 episodes (one per cluster)
    actual_episodes = len(episodes)

    print(f"\nExpected episodes: {expected_episodes}")
    print(f"Actual episodes: {actual_episodes}")
    print(f"Durations > 1: {sum(1 for ep in episodes if ep.episode_length() > 1)}")

    issues = []
    if all_duration_1 and len(episodes) > 1:
        issues.append("ISSUE: All episodes have duration=1 (merging not working)")
    if actual_episodes != expected_episodes:
        issues.append(f"ISSUE: Expected {expected_episodes} episodes, got {actual_episodes}")

    return issues


def test_type_classification():
    """Test if types are properly assigned (not always same type)"""
    print("\n" + "="*80)
    print("TEST 2: Type Classification Diversity")
    print("="*80)

    # Create data with different patterns
    delta_L = np.array([0.1]*200, dtype=float)

    # Sudden peak (index 50) - should be 'ABRUPT'
    delta_L[50] = 2.5

    # Gradual increase (indices 100-110)
    delta_L[100:111] = np.linspace(0.3, 0.8, 11)

    # Incremental trend (indices 150-180)
    delta_L[150:181] = np.linspace(0.1, 0.7, 31)

    similarity = np.ones(200) * 0.5
    similarity[50] = 0.05
    similarity[100:111] = np.linspace(0.5, 0.2, 11)
    similarity[150:181] = np.linspace(0.5, 0.1, 31)

    drift_indices = [50, 100, 102, 104, 106, 108, 110, 150, 160, 170, 180]

    agg = DriftAggregatorV2(
        delta_L_history=delta_L,
        similarity_history=similarity,
        drift_indices=drift_indices,
        window_size=50,
        warm_up_windows=2,
        merge_gap=5,
        cooldown=10
    )

    episodes = agg.get_merged_episodes()
    types = agg.get_type_counts()

    print(f"Episodes generated: {len(episodes)}")
    print(f"Type distribution: {types}")
    print(f"Dominant type: {agg.get_dominant_type()}")

    if episodes:
        unique_types = set(ep.dominant_type for ep in episodes)
        print(f"Unique types in episodes: {unique_types}")
        print(f"Episode details:")
        for i, ep in enumerate(episodes):
            print(f"  Ep {i+1}: type={ep.dominant_type}, votes={ep.type_votes}, z_scores={ep.z_scores[:3]}")

    issues = []
    if len(episodes) > 1:
        unique_types = set(ep.dominant_type for ep in episodes)
        if len(unique_types) == 1:
            issues.append(f"ISSUE: All episodes classified as '{list(unique_types)[0]}' (type classifier stuck)")

    return issues


def check_diagnostics_validity():
    """Check if diagnostics JSON from bikes dataset is valid"""
    print("\n" + "="*80)
    print("TEST 3: Diagnostics JSON Validity")
    print("="*80)

    json_path = Path("experiments/results/bikes_w50_t0.4_a2.0/report_debug.json")

    if not json_path.exists():
        print(f"DIAGNOSTIC FILE NOT FOUND: {json_path}")
        return ["ISSUE: No diagnostic file to validate"]

    try:
        with open(json_path) as f:
            data = json.load(f)

        print(f"JSON valid: Yes")
        print(f"Top-level keys: {list(data.keys())}")

        # Check KPI values
        print(f"\nKPI Values:")
        print(f"  total_instances: {data.get('total_instances')} (type: {type(data.get('total_instances')).__name__})")
        print(f"  raw_drift_count: {data.get('raw_drift_count')} (type: {type(data.get('raw_drift_count')).__name__})")
        print(f"  merged_episodes_count: {data.get('merged_episodes_count')} (type: {type(data.get('merged_episodes_count')).__name__})")
        print(f"  drift_rate_percent: {data.get('drift_rate_percent')} (type: {type(data.get('drift_rate_percent')).__name__})")

        # Check for garbage values
        issues = []
        for key in ['total_instances', 'raw_drift_count', 'merged_episodes_count']:
            val = data.get(key)
            if val is None or (isinstance(val, str) and len(val) > 100):
                issues.append(f"ISSUE: {key} has suspicious value: {val}")
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                issues.append(f"ISSUE: {key} is NaN/Inf: {val}")

        if not issues:
            print(f"\nKPI values look valid")

        return issues

    except json.JSONDecodeError as e:
        return [f"ISSUE: Invalid JSON: {str(e)}"]
    except Exception as e:
        return [f"ISSUE: Error reading diagnostics: {str(e)}"]


def check_html_report():
    """Check HTML report for common issues"""
    print("\n" + "="*80)
    print("TEST 4: HTML Report Content Quality")
    print("="*80)

    html_path = Path("experiments/results/bikes_w50_t0.4_a2.0/report_en.html")

    if not html_path.exists():
        print(f"HTML REPORT NOT FOUND: {html_path}")
        return ["ISSUE: No HTML report to validate"]

    try:
        with open(html_path, encoding='utf-8') as f:
            html_content = f.read()

        print(f"HTML file size: {len(html_content)} bytes")

        issues = []

        # Check for dev content leaks
        if "save_diagnostic_log" in html_content:
            issues.append("ISSUE: Dev code 'save_diagnostic_log' found in HTML")
        if "TODO" in html_content or "FIXME" in html_content:
            issues.append("ISSUE: Dev comments (TODO/FIXME) found in HTML")
        if "import sys" in html_content or "import json" in html_content:
            issues.append("ISSUE: Python code import statements found in HTML")

        # Check for local file paths
        if "file:///" in html_content:
            count = html_content.count("file:///")
            issues.append(f"ISSUE: {count} local file:/// paths found in HTML (should be sanitized)")

        # Check for KPI values
        if "Total Instances" in html_content:
            # Find the value after label
            idx = html_content.find("Total Instances")
            snippet = html_content[idx:idx+200]
            print(f"\nTotal Instances section found:")
            print(f"  ...{snippet[:100]}...")

        # Check for graph descriptions
        if "Delta L" in html_content:
            print(f"Delta L section: Found")
        else:
            issues.append("ISSUE: Delta L section not found in HTML")

        if not issues:
            print(f"\nHTML content looks valid")

        return issues

    except Exception as e:
        return [f"ISSUE: Error reading HTML: {str(e)}"]


def main():
    print("\n" + "="*80)
    print("PHASE 6 DIAGNOSTIC TEST SUITE")
    print("Identifying all reported issues")
    print("="*80)

    all_issues = []

    # Run all tests
    all_issues.extend(test_episode_duration_and_merging())
    all_issues.extend(test_type_classification())
    all_issues.extend(check_diagnostics_validity())
    all_issues.extend(check_html_report())

    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF ISSUES FOUND")
    print("="*80)

    if not all_issues:
        print("✓ No critical issues found!")
    else:
        print(f"\n{len(all_issues)} issues identified:\n")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")

    return len(all_issues)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(1 if exit_code > 0 else 0)
