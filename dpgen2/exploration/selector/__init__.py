from .conf_filter import (
    ConfFilter,
    ConfFilters,
)
from .conf_selector import (
    ConfSelector,
)
from .conf_selector_frame import (
    ConfSelectorFrames,
)
from .distance_conf_filter import (
    BoxLengthFilter,
    BoxSkewnessConfFilter,
    DistanceConfFilter,
)

conf_filter_styles = {
    "distance": DistanceConfFilter,
    "box_skewness": BoxSkewnessConfFilter,
    "box_length": BoxLengthFilter,
}
