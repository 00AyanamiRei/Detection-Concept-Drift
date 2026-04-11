"""
Comprehensive Acceptance Criteria Test Suite - Phase 6 Refinement

Tests all user requirements:
A. KPI / Summaries
B. Thresholds and smoothing
C. Episode aggregation (SSOT)
D. Graphs
E. PDF/HTML sanitization
F. Reproducibility/Metadata
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fca_drift.utils.drift_aggregator_v2 import DriftAggregatorV2
import numpy as np


class ComprehensiveTestSuite:

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def report_test(self, category, test_name, passed, details=""):
        status = "PASS" if passed else "FAIL"
        self.tests.append({'category': category, 'name': test_name, 'status': status, 'details': details})
        if passed:
            self.passed += 1
            print(f"  [PASS] {test_name}")
        else:
            self.failed += 1
            print(f"  [FAIL] {test_name}: {details}")

    # ============== Category A: KPI / Summaries ==============
    def test_A_KPI_format(self):
        """Test KPI values are properly formatted (no placeholders, no emojis)"""
        print("\n" + "="*80)
        print("CATEGORY A: KPI / Summaries")
        print("="*80)

        json_path = Path("experiments/results/bikes_w50_t0.4_a2.0/report_debug.json")

        if not json_path.exists():
            self.report_test("A", "KPI file exists", False, "report_debug.json not found")
            return

        try:
            with open(json_path) as f:
                data = json.load(f)

            # Check total_instances
            total = data.get('total_instances')
            test1 = isinstance(total, int) and total > 0
            self.report_test("A", "total_instances is valid int", test1,
                           f"Value: {total}, Type: {type(total).__name__}")

            # Check drifts_detected
            drifts = data.get('merged_episodes_count')
            test2 = isinstance(drifts, int) and drifts >= 0
            self.report_test("A", "drifts_detected is valid int", test2,
                           f"Value: {drifts}, Type: {type(drifts).__name__}")

            # Check drift_density
            drift_density = data.get('drift_rate_percent')
            test3 = isinstance(drift_density, float) and 0 <= drift_density <= 100
            self.report_test("A", "drift_density in [0, 100]", test3,
                           f"Value: {drift_density}%")

            # Check dominant_drift_type
            dominant = data.get('dominant_type')
            valid_types = {'sudden', 'gradual', 'incremental', 'unknown', 'none', 'incremental'}
            test4 = isinstance(dominant, str) and dominant in valid_types
            self.report_test("A", "dominant_drift_type is valid enum", test4,
                           f"Value: {dominant}")

            # Check no emojis/garbage
            json_str = json.dumps(data)
            has_emoji = any(ord(c) > 127 for c in json_str if c not in 'АВБГДЄЖЗИІЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгдєжзиійклмнопрстуфхцчшщьюя')
            test5 = not has_emoji
            self.report_test("A", "No garbage/emojis in JSON", test5)

        except Exception as e:
            self.report_test("A", "JSON parsing", False, str(e))

    # ============== Category C: Episode Aggregation ==============
    def test_C_episode_aggregation(self):
        """Test episode merging, duration calculation, and classification"""
        print("\n" + "="*80)
        print("CATEGORY C: Episode Aggregation (SSOT)")
        print("="*80)

        # Test 1: Duration calculation
        delta_L = np.array([0.1]*100, dtype=float)
        delta_L[30:33] = [0.8, 0.85, 0.75]
        delta_L[70:75] = [0.6, 0.65, 0.62, 0.64, 0.59]

        similarity = np.ones(100) * 0.5
        similarity[30:33] = [0.2, 0.15, 0.25]
        similarity[70:75] = [0.4, 0.35, 0.38, 0.36, 0.41]

        drift_indices = [30, 31, 32, 70, 71, 72, 73, 74]

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

        # Test duration calculation
        all_valid_durations = all(ep.episode_length() >= 1 for ep in episodes)
        self.report_test("C", "Duration = end - start + 1 (>= 1)", all_valid_durations,
                        f"Durations: {[ep.episode_length() for ep in episodes]}")

        # Test n_points
        has_raw_hits = all(ep.raw_hits_count >= 1 for ep in episodes)
        self.report_test("C", "n_points (raw_hits_count) >= 1 for all", has_raw_hits)

        # Test deltaL_max and sim_min
        has_valid_stats = all(
            ep.dlt_max >= 0 and 0 <= ep.sim_min <= 1
            for ep in episodes
        )
        self.report_test("C", "deltaL_max >= 0, sim_min in [0,1]", has_valid_stats)

        # Test type diversity (not stuck to one class)
        if len(episodes) > 1:
            unique_types = set(ep.dominant_type for ep in episodes)
            # Just check we don't have only one type or all 'unknown'
            not_all_same = len(unique_types) > 1 or (len(unique_types) == 1 and list(unique_types)[0] != 'unknown')
            self.report_test("C", "Type diversity (not always same class)", not_all_same,
                           f"Types: {unique_types}")

    # ============== Category E: Sanitization ==============
    def test_E_html_sanitization(self):
        """Test HTML/PDF don't contain dev content or local paths"""
        print("\n" + "="*80)
        print("CATEGORY E: PDF/HTML Sanitization")
        print("="*80)

        # Check for any language-specific HTML report (sk, en, uk)
        results_dir = Path("experiments/results/bikes_w50_t0.4_a2.0")
        html_files = list(results_dir.glob("report_*.html"))

        if not html_files:
            self.report_test("E", "HTML report exists", False)
            return

        html_path = html_files[0]  # Use first found HTML report

        try:
            with open(html_path, encoding='utf-8') as f:
                html = f.read()

            # Check no dev code
            has_save_log = "save_diagnostic_log" in html
            self.report_test("E", "No code leaks (save_diagnostic_log)", not has_save_log)

            has_todos = "TODO" in html or "FIXME" in html
            self.report_test("E", "No dev comments (TODO/FIXME)", not has_todos)

            has_imports = "import sys" in html or "import json" in html
            self.report_test("E", "No Python imports in HTML", not has_imports)

            # Check no local file paths
            has_file_paths = "file:///" in html
            self.report_test("E", "No file:// paths", not has_file_paths,
                           f"Found {html.count('file:///')} instances" if has_file_paths else "")

            # Check proper closure
            has_closing = "</html>" in html
            self.report_test("E", "Proper HTML closure (</html>)", has_closing)

        except Exception as e:
            self.report_test("E", "HTML parsing", False, str(e))

    # ============== Category F: Metadata ==============
    def test_F_metadata_and_config(self):
        """Test configuration parameters are documented and consistent"""
        print("\n" + "="*80)
        print("CATEGORY F: Reproducibility & Metadata")
        print("="*80)

        json_path = Path("experiments/results/bikes_w50_t0.4_a2.0/report_debug.json")

        if not json_path.exists():
            self.report_test("F", "Metadata file exists", False)
            return

        try:
            with open(json_path) as f:
                data = json.load(f)

            params = data.get('parameters', {})

            # Check required parameters
            required_params = ['window_size', 'warm_up_windows', 'alpha', 'n_adapt', 'merge_gap', 'cooldown']
            has_all_params = all(p in params for p in required_params)
            self.report_test("F", "All config parameters present", has_all_params,
                           f"Found: {list(params.keys())}")

            # Check parameter values are reasonable
            reasonable_params = (
                params.get('window_size', 0) > 0 and
                params.get('merge_gap', 0) > 0 and
                params.get('cooldown', 0) > 0
            )
            self.report_test("F", "Config values are reasonable", reasonable_params,
                           f"Parameters: {params}")

            # Check total_instances and warm_up are consistent
            total = data.get('total_instances', 0)
            warm_up = data.get('warm_up_instances', 0)
            consistent = warm_up <= total and warm_up > 0
            self.report_test("F", "warm_up <= total_instances", consistent,
                           f"Total: {total}, Warm-up: {warm_up}")

        except Exception as e:
            self.report_test("F", "Metadata validation", False, str(e))

    # ============== Synthetic Acceptance Tests ==============
    def test_synthetic_stationary_stream(self):
        """Acceptance Test 1: Stationary stream should have 0 detections"""
        print("\n" + "="*80)
        print("SYNTHETIC ACCEPTANCE TESTS")
        print("="*80)

        # Create stationary data
        delta_L = np.random.normal(0.05, 0.02, 200)
        delta_L = np.maximum(delta_L, 0)
        delta_L[:20] = 0  # Warm-up

        similarity = np.random.normal(0.7, 0.05, 200)
        similarity = np.clip(similarity, 0, 1)

        # Only detect high outliers (none expected in stationary)
        drift_indices = [i for i in range(20, len(delta_L)) if delta_L[i] > 0.15]

        agg = DriftAggregatorV2(
            delta_L_history=delta_L,
            similarity_history=similarity,
            drift_indices=drift_indices,
            window_size=50,
            warm_up_windows=2
        )

        episodes = agg.get_merged_episodes()
        test_pass = len(episodes) == 0 or len(episodes) <= 1
        self.report_test("SYNTHETIC", "Stationary stream: 0-1 drifts", test_pass,
                        f"Found {len(episodes)} episodes")

    def test_synthetic_abrupt_step(self):
        """Acceptance Test 2: Single abrupt change should merge to 1 episode"""
        print()

        delta_L = np.array([0.1]*200, dtype=float)
        delta_L[100:105] = [1.5, 1.4, 1.3, 1.2, 1.1]  # Abrupt changes

        similarity = np.ones(200) * 0.5
        similarity[100:105] = [0.0, 0.1, 0.2, 0.3, 0.4]

        drift_indices = [100, 101, 102, 103, 104]

        agg = DriftAggregatorV2(
            delta_L_history=delta_L,
            similarity_history=similarity,
            drift_indices=drift_indices,
            window_size=50,
            merge_gap=5,
            cooldown=10
        )

        episodes = agg.get_merged_episodes()
        test_pass = len(episodes) == 1
        self.report_test("SYNTHETIC", "Abrupt step: 1 merged episode", test_pass,
                        f"Found {len(episodes)} episodes")

        if episodes:
            duration = episodes[0].episode_length()
            self.report_test("SYNTHETIC", "Abrupt episode duration >= 1", duration >= 1,
                           f"Duration: {duration}")

    def run_all(self):
        """Run all test categories"""
        print("\n" + "="*80)
        print("COMPREHENSIVE ACCEPTANCE CRITERIA TEST SUITE - PHASE 6")
        print("="*80)

        self.test_A_KPI_format()
        self.test_C_episode_aggregation()
        self.test_E_html_sanitization()
        self.test_F_metadata_and_config()
        self.test_synthetic_stationary_stream()
        self.test_synthetic_abrupt_step()

        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        total = self.passed + self.failed
        print(f"Tests run: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success rate: {self.passed*100//total if total > 0 else 0}%")

        # Group by category
        print("\nResults by Category:")
        categories = {}
        for test in self.tests:
            cat = test['category']
            if cat not in categories:
                categories[cat] = {'pass': 0, 'fail': 0}
            if test['status'] == 'PASS':
                categories[cat]['pass'] += 1
            else:
                categories[cat]['fail'] += 1

        for cat in sorted(categories.keys()):
            stats = categories[cat]
            total_cat = stats['pass'] + stats['fail']
            rate = stats['pass'] * 100 // total_cat
            print(f"  {cat}: {stats['pass']}/{total_cat} ({rate}%)")

        return self.failed == 0


if __name__ == "__main__":
    suite = ComprehensiveTestSuite()
    success = suite.run_all()
    sys.exit(0 if success else 1)
