from .report import (
    ExplorationReport,
)
from .report_trust_levels import (
    ExplorationReportTrustLevels,
)
from .report_adaptive_lower import (
    ExplorationReportAdaptiveLower,
)

conv_styles = {
    "fixed-levels": ExplorationReportTrustLevels,
    "adaptive-lower": ExplorationReportAdaptiveLower,
}
