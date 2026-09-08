import re
from pathlib import (
    Path,
)

from dpgen2.fp import (
    fp_styles,
)


def test_feature_matrix_matches_registered_fp_styles():
    """Keep every submit-time FP backend discoverable in the feature matrix."""
    feature_doc = Path("docs/features.md").read_text()
    for style in fp_styles:
        assert f"`{style}`" in feature_doc


def test_feature_matrix_local_links_exist():
    """Prevent source and example links in the support table from going stale."""
    feature_path = Path("docs/features.md")
    feature_doc = feature_path.read_text()
    local_links = re.findall(r"\[[^]]+\]\((?!https?://)([^)#]+)", feature_doc)
    assert local_links
    for link in local_links:
        assert (feature_path.parent / link).resolve().exists(), link
