from utils.context import dpgen2
import numpy as np
import unittest, json, shutil, os
import random
import tempfile
import dpdata
from pathlib import Path

from dpgen2.utils import (
    dflow_config
)
from dflow import config, s3_config

class TestDflowConfig(unittest.TestCase):
    def test_config(self):
        config_data = {
	    "host" : "foo",
	    "s3_endpoint" : "bar",
	    "k8s_api_server" : "tar",
            "token" : "bula",
        }
        dflow_config(config_data)
        self.assertEqual(config['host'], 'foo')
        self.assertEqual(s3_config['endpoint'], 'bar')
        self.assertEqual(config['k8s_api_server'], 'tar')
        self.assertEqual(config['token'], 'bula')
        
    def test_none(self):
        config_data = {
	    "host" : "foo",
	    "s3_endpoint" : None,
	    "k8s_api_server" : None,
            "token" : "bula",
        }
        dflow_config(config_data)
        self.assertEqual(config['host'], 'foo')
        self.assertEqual(s3_config['endpoint'], None)
        self.assertEqual(config['k8s_api_server'], None)
        self.assertEqual(config['token'], 'bula')

    def test_empty(self):
        config_data = {
        }
        dflow_config(config_data)
        self.assertEqual(config['host'], None)
        self.assertEqual(s3_config['endpoint'], None)
        self.assertEqual(config['k8s_api_server'], None)
        self.assertEqual(config['token'], None)
        
