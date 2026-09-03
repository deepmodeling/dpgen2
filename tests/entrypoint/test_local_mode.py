import os
import unittest
from unittest import (
    mock,
)

import dflow

# isort: off
from .context import (
    dpgen2,
)
from dpgen2.entrypoint.common import (
    global_config_workflow,
)

# isort: on


class TestLocalMode(unittest.TestCase):
    @mock.patch("dpgen2.entrypoint.common.bohrium_config_from_dict")
    @mock.patch.dict(os.environ, {"DFLOW_MODE": "debug"}, clear=True)
    def test_dflow_mode_debug_skips_bohrium_configuration(self, mocked_bohrium):
        previous_mode = dflow.config.get("mode")
        self.addCleanup(dflow.config.__setitem__, "mode", previous_mode)

        global_config_workflow(
            {
                # Invalid on purpose: debug mode must return before reading it.
                "bohrium_config": {"username": "not-used"},
            }
        )

        self.assertEqual(dflow.config["mode"], "debug")
        mocked_bohrium.assert_not_called()

    @mock.patch("dpgen2.entrypoint.common.bohrium_config_from_dict")
    @mock.patch.dict(os.environ, {"DFLOW_DEBUG": "1"}, clear=True)
    def test_legacy_dflow_debug_skips_bohrium_configuration(self, mocked_bohrium):
        previous_mode = dflow.config.get("mode")
        self.addCleanup(dflow.config.__setitem__, "mode", previous_mode)

        global_config_workflow({"bohrium_config": {"username": "not-used"}})

        self.assertEqual(dflow.config["mode"], "debug")
        mocked_bohrium.assert_not_called()
