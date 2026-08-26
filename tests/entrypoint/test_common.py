import json
import tempfile
import unittest
from pathlib import (
    Path,
)

# isort: off
from .context import (
    dpgen2,
)
from dpgen2.entrypoint.common import (
    load_config,
)

# isort: on


class TestLoadConfig(unittest.TestCase):
    def test_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps({"type_map": ["H", "O"]}))

            self.assertEqual(load_config(path), {"type_map": ["H", "O"]})

    def test_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.yaml"
            path.write_text(
                """# YAML permits comments and unquoted mapping keys.
type_map:
  - H
  - O
explore:
  fatal_at_max: false
"""
            )

            self.assertEqual(
                load_config(path),
                {
                    "type_map": ["H", "O"],
                    "explore": {"fatal_at_max": False},
                },
            )

    def test_document_root_must_be_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.yml"
            path.write_text("- not\n- a\n- mapping\n")

            with self.assertRaisesRegex(ValueError, "root must be a mapping"):
                load_config(path)
