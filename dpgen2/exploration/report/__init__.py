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

conv_styles = {
    "fixed-levels": ExplorationReportTrustLevelsRandom,
    "fixed-levels-max-select": ExplorationReportTrustLevelsMax,
    "adaptive-lower": ExplorationReportAdaptiveLower,
}
