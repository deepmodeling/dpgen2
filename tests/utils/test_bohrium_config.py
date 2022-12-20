from utils.context import dpgen2
import numpy as np
import unittest, json, shutil, os
import random
import tempfile
import dpdata
from pathlib import Path

from dpgen2.utils import (
    bohrium_config_from_dict,
)
import dflow
from dflow import config, s3_config
from dflow.plugins import bohrium


import pytest
@pytest.mark.server(url="/account/login", response={"code": 0, "data": {"token": "abc"}}, method="POST")
@pytest.mark.server(url="/brm/v1/storage/token", response={"code": 0, "data": {"token": "abc", "path": "/", "sharePath": "/", "userSharePath": "/"}}, method="GET")
def test_handler_responses():
    bohrium_config = {
        "host" : "666",
        "k8s_api_server" : "777",
        "username": "foo",
        "password": "bar",
        "project_id": 10086,
        "repo_key": "tar",
        "storage_client" : "dflow.plugins.bohrium.TiefblueClient"
    }
    bohrium.config["bohrium_url"] = "http://localhost:5000"
    bohrium_config_from_dict(bohrium_config)
    assert(config["host"] == "666")
    assert(config["k8s_api_server"] == "777")
    assert(bohrium.config["username"] == "foo")
    assert(bohrium.config["password"] == "bar")
    assert(bohrium.config["project_id"] == "10086")
    assert(s3_config["repo_key"] == "tar")
    assert(isinstance(s3_config["storage_client"], dflow.plugins.bohrium.TiefblueClient))


# @unittest.skipIf(True, "dflow requires a real bohrium account to instantiate a storage client")
# class TestBohriumConfig(unittest.TestCase):
#     def test_config(self):
#         bohrium_config = {
#             "host" : "666",
#             "k8s_api_server" : "777",
#             "username": "foo",
#             "password": "bar",
#             "project_id": 10086,
#             "repo_key": "tar",
#             "storage_client" : "dflow.plugins.bohrium.TiefblueClient"
#         }
#         bohrium_config_from_dict(bohrium_config)
#         self.assertEqual(config["host"], "666")
#         self.assertEqual(config["k8s_api_server"], "777")
#         self.assertEqual(bohrium.config["username"], "foo")
#         self.assertEqual(bohrium.config["password"], "bar")
#         self.assertEqual(bohrium.config["project_id"], "10086")
#         self.assertEqual(s3_config["repo_key"], "tar")
#         self.assertTrue(isinstance(s3_config["storage_client"],
#                                    dflow.plugins.bohrium.TiefblueClient))
