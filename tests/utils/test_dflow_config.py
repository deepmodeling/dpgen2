import json
import os
import random
import shutil
import tempfile
import unittest
from pathlib import (
    Path,
)

import dpdata
import numpy as np
from dflow.config import (
    config,
    s3_config,
)

# isort: off
from .context import (
    dpgen2,
)
from dpgen2.utils import (
    dflow_config,
    dflow_s3_config,
)

# isort: on


class TestDflowConfig(unittest.TestCase):
    def test_config(self):
        config_data = {
            "host": "foo",
            "s3_endpoint": "bar",
            "k8s_api_server": "tar",
            "token": "bula",
        }
        dflow_config(config_data)
        self.assertEqual(config["host"], "foo")
        self.assertEqual(s3_config["endpoint"], "bar")
        self.assertEqual(config["k8s_api_server"], "tar")
        self.assertEqual(config["token"], "bula")

    def test_none(self):
        config_data = {
            "host": "foo",
            "s3_endpoint": None,
            "k8s_api_server": None,
            "token": "bula",
        }
        dflow_config(config_data)
        self.assertEqual(config["host"], "foo")
        self.assertEqual(s3_config["endpoint"], None)
        self.assertEqual(config["k8s_api_server"], None)
        self.assertEqual(config["token"], "bula")

    def test_empty(self):
        config_data = {
            "host": None,
            "s3_endpoint": None,
            "k8s_api_server": None,
            "token": None,
        }
        dflow_config(config_data)
        self.assertEqual(config["host"], None)
        self.assertEqual(s3_config["endpoint"], None)
        self.assertEqual(config["k8s_api_server"], None)
        self.assertEqual(config["token"], None)

    def test_s3_config(self):
        config_data = {
            "endpoint": "bar",
        }
        dflow_s3_config(config_data)
        self.assertEqual(s3_config["endpoint"], "bar")

    def test_none(self):
        config_data = {
            "endpoint": None,
        }
        dflow_s3_config(config_data)
        self.assertEqual(s3_config["endpoint"], None)
