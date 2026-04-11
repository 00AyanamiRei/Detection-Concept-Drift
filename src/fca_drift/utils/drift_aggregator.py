"""
Unified drift type aggregation and normalization module

Purpose: Single source of truth for drift type statistics
- Normalizes drift type names across languages
- Computes dominant type and badge
- Generates diagnostic logs
"""
from typing import Dict, List, Tuple, Optional
from collections import Counter
import json
from pathlib import Path


class DriftTypeNormalizer:
    """Normalize drift type names to canonical forms"""

    # Map of all possible representations to canonical form
    TYPE_MAPPINGS = {
        # Canonical forms (lowercase)
        'sudden': 'sudden',
        'gradual': 'gradual',
        'incremental': 'incremental',
        'none': 'none',

        # English variants
        'abrupt': 'sudden',
        'sharp': 'sudden',
        'rapid': 'sudden',
        'slow': 'gradual',
        'gradational': 'gradual',
        'progressive': 'gradual',
        'incremental_change': 'incremental',

        # Slovak variants (náhly, postupný, inkrementálny)
        'náhly': 'sudden',
        'prudky': 'sudden',
        'postúpny': 'gradual',
        'pozvolny': 'gradual',
        'inkrementálny': 'incremental',
        'postupny': 'gradual',

        # Ukrainian variants (раптовий, поступовий, інкрементальний)
        'раптовий': 'sudden',
        'рaptовий': 'sudden',
        'поступовий': 'gradual',
        'постійний': 'gradual',
        'інкрементальний': 'incremental',
        'inkrementalnyi': 'incremental',

        # Capitalized variants
        'Sudden': 'sudden',
        'Gradual': 'gradual',
        'Incremental': 'incremental',
        'Náhly': 'sudden',
        'Postupný': 'gradual',
    }

    @staticmethod
    def normalize(drift_type: str) -> str:
        """
        Normalize drift type string to canonical form (lowercase)

        Args:
            drift_type: Any representation of drift type

        Returns:
            Canonical form: one of {'sudden', 'gradual', 'incremental', 'none'}
        """
        if not drift_type:
            return 'none'

        # Clean up
        normalized = drift_type.strip().lower()

        # Direct lookup
        if normalized in DriftTypeNormalizer.TYPE_MAPPINGS:
            return DriftTypeNormalizer.TYPE_MAPPINGS[normalized]

        # Partial match fallback
        for key, canonical in DriftTypeNormalizer.TYPE_MAPPINGS.items():
            if key in normalized or normalized in key:
                return canonical

        # Default to the input (lowercased) if unknown
        return normalized


class DriftTypeAggregator:
    """Aggregate and analyze drift types across a run"""

    def __init__(self, drift_indices: List[int], drift_types: Dict[int, str]):
        """
        Initialize with drift events

        Args:
            drift_indices: List of instance indices where drift occurred
            drift_types: Dict mapping instance index to drift type string
        """
        self.drift_indices = drift_indices
        self.drift_types = drift_types

        # Compute aggregations
        self._compute_statistics()

    def _compute_statistics(self):
        """Compute unified statistics"""
        # Normalize all types
        normalized_types = []
        for idx in self.drift_indices:
            raw_type = self.drift_types.get(idx, 'none')
            norm_type = DriftTypeNormalizer.normalize(raw_type)
            normalized_types.append(norm_type)

        # Count by type
        self.counts = Counter(normalized_types)
        self.total_drifts = len(self.drift_indices)

        # Dominant type (or 'none' if no drifts)
        if self.total_drifts == 0:
            self.dominant_type = 'none'
        else:
            self.dominant_type = self.counts.most_common(1)[0][0]

        # Percentages
        self.percentages = {}
        for dtype in ['sudden', 'gradual', 'incremental']:
            if self.total_drifts > 0:
                self.percentages[dtype] = self.counts.get(dtype, 0) / self.total_drifts
            else:
                self.percentages[dtype] = 0.0

        self.share_sudden = self.percentages.get('sudden', 0.0)
        self.share_gradual = self.percentages.get('gradual', 0.0)
        self.share_incremental = self.percentages.get('incremental', 0.0)

    def get_dominant_type(self) -> str:
        """Return dominant drift type"""
        return self.dominant_type

    def get_counts(self) -> Dict[str, int]:
        """Return counts by type"""
        return dict(self.counts)

    def get_total_by_type(self, dtype: str) -> int:
        """Get count for specific type"""
        return self.counts.get(dtype, 0)

    def to_dict(self) -> Dict:
        """Export as diagnostic dict"""
        return {
            'total_drifts': self.total_drifts,
            'dominant_type': self.dominant_type,
            'counts_by_type': dict(self.counts),
            'percentages_by_type': {
                'sudden': round(self.share_sudden, 3),
                'gradual': round(self.share_gradual, 3),
                'incremental': round(self.share_incremental, 3),
            }
        }


class ConclusionBuilder:
    """Build conclusion based on drift statistics"""

    # Badge thresholds (configurable)
    BADGE_THRESHOLDS = {
        'sudden_critical': 0.5,      # >= 50% sudden → 🔴
        'sudden_warning': 0.2,       # >= 20% sudden → 🟠 (unless dominated by other)
        'gradual_warning': 0.5,      # >= 50% gradual → 🟠
        'none_badge': '✓',           # No drift
    }

    # Conclusion templates by language
    TEMPLATES = {
        'sk': {
            'sudden': {
                'badge': '🔴',
                'text': 'Prudky drift',
                'final': 'Zisteny PRUDKY drift v {count} bodoch. Vyžaduje sa OKAMZITA intervencia!'
            },
            'gradual': {
                'badge': '🟠',
                'text': 'Pozvolny drift',
                'final': 'Zisteny POZVOLNY drift v {count} bodoch. Odporuca sa monitoring a postupne preskupenie.'
            },
            'incremental': {
                'badge': '🟡',
                'text': 'Inkrementalny drift',
                'final': 'Zisteny INKREMENTALNY drift v {count} bodoch. Odporuca sa monitoring.'
            },
            'none': {
                'badge': '✓',
                'text': 'Zaden drift',
                'final': 'Zaden drift bol detegovany. System je stabilny.'
            },
        },
        'uk': {
            'sudden': {
                'badge': '🔴',
                'text': 'Раптовий дрейф',
                'final': 'Виявлено РАПТОВИЙ дрейф в {count} точках. Потрібна НЕГАЙНА дія!'
            },
            'gradual': {
                'badge': '🟠',
                'text': 'Поступовий дрейф',
                'final': 'Виявлено ПОСТУПОВИЙ дрейф в {count} точках. Рекомендується моніторинг та поступна переадаптація.'
            },
            'incremental': {
                'badge': '🟡',
                'text': 'Інкрементальний дрейф',
                'final': 'Виявлено ІНКРЕМЕНТАЛЬНИЙ дрейф в {count} точках. Рекомендується моніторинг.'
            },
            'none': {
                'badge': '✓',
                'text': 'Без дрейфу',
                'final': 'Дрейф не виявлено. Система стабільна.'
            },
        },
        'en': {
            'sudden': {
                'badge': '🔴',
                'text': 'Sudden Drift',
                'final': 'SUDDEN drift detected in {count} points. IMMEDIATE action required!'
            },
            'gradual': {
                'badge': '🟠',
                'text': 'Gradual Drift',
                'final': 'GRADUAL drift detected in {count} points. Monitoring and gradual retraining recommended.'
            },
            'incremental': {
                'badge': '🟡',
                'text': 'Incremental Drift',
                'final': 'INCREMENTAL drift detected in {count} points. Monitoring recommended.'
            },
            'none': {
                'badge': '✓',
                'text': 'No Drift',
                'final': 'No drift detected. System is stable.'
            },
        },
    }

    @staticmethod
    def build_conclusion(aggregator: DriftTypeAggregator, language: str = 'sk') -> Tuple[str, str, str]:
        """
        Build conclusion badge, text, and full statement

        Args:
            aggregator: DriftTypeAggregator with statistics
            language: 'sk', 'uk', or 'en'

        Returns:
            (badge, conclusion_text, final_conclusion)
            Example: ('🔴', 'Sudden Drift', 'SUDDEN drift detected...')
        """
        dominant = aggregator.get_dominant_type()
        count = aggregator.get_total_by_type(dominant)

        # Get templates for language
        templates = ConclusionBuilder.TEMPLATES.get(language, ConclusionBuilder.TEMPLATES['en'])

        # Handle 'unknown' type by defaulting to 'incremental' (conservative drift assumption)
        template_key = dominant if dominant in templates else 'incremental'
        conclusion_template = templates[template_key]

        badge = conclusion_template['badge']
        conclusion_text = conclusion_template['text']
        final_conclusion = conclusion_template['final']

        # Add count to final conclusion if there are drifts
        if dominant != 'none' and count > 0:
            final_conclusion = final_conclusion.format(count=count)

        return badge, conclusion_text, final_conclusion

    @staticmethod
    def save_diagnostic_log(
        aggregator: DriftTypeAggregator,
        output_path: Path,
        dataset_id: str,
        run_id: str,
        window_size: int,
        theta: float,
        alpha: float
    ):
        """Save diagnostic log for verification"""
        diagnostic = {
            'run_id': run_id,
            'dataset_id': dataset_id,
            'timestamp': str(Path(output_path).stat().st_mtime),
            'parameters': {
                'window_size': window_size,
                'theta': theta,
                'alpha': alpha,
            },
            'statistics': aggregator.to_dict(),
        }

        debug_path = output_path.parent / 'report_debug.json'
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(diagnostic, f, indent=2, ensure_ascii=False)
