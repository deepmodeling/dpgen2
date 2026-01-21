from .report import (
    ExplorationReport,
)
from .report_adaptive_lower import (
    ExplorationReportAdaptiveLower,
)
from .report_trust_levels_max import (
    ExplorationReportTrustLevelsMax,
)
from .report_trust_levels_random import (
    ExplorationReportTrustLevelsRandom,
)
from .report_trust_levels_spin import (
    ExplorationReportTrustLevelsSpin,
)

conv_styles = {
    "fixed-levels": ExplorationReportTrustLevelsRandom,
    "fixed-levels-max-select": ExplorationReportTrustLevelsMax,
    "fixed-levels-max-select-spin": ExplorationReportTrustLevelsSpin,
    "adaptive-lower": ExplorationReportAdaptiveLower,
}
