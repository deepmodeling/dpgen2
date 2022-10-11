import os
import numpy as np
import unittest

from dflow import (
    InputParameter,
    OutputParameter,
    Inputs,
    InputArtifact,
    Outputs,
    OutputArtifact,
    Workflow,
    Step,
    Steps,
    upload_artifact,
    download_artifact,
    S3Artifact,
    argo_range
)
from dflow.python import (
    PythonOPTemplate,
    OP,
    OPIO,
    OPIOSign,
    Artifact,
)

import time, shutil, json, jsonpickle
from typing import Set, List, Tuple
from pathlib import Path
try:
    from context import dpgen2
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from mocked_ops import (
    MockedSelectConfs,
    MockedConfSelector,
    MockedExplorationReport,
)
from dpgen2.op.select_confs import SelectConfs

class TestMockedSelectConfs(unittest.TestCase):
    def setUp(self): 
        self.conf_selector = MockedConfSelector()
        self.traj_fmt = 'foo'
        self.type_map = []
        self.trajs = [Path('traj.foo'), Path('traj.bar')]
        self.model_devis = [Path('md.foo'), Path('md.bar')]

    def tearDown(self):
        for ii in ['conf.0', 'conf.1']:
            ii=Path(ii)
            if ii.is_file():
                os.remove(ii)
    
    def test(self):
        op = MockedSelectConfs()
        out = op.execute(OPIO({
            'conf_selector': self.conf_selector,
            'traj_fmt': self.traj_fmt,
            'type_map' : self.type_map,
            'trajs' : self.trajs,
            'model_devis' : self.model_devis,
        }))
        confs = out['confs']
        report = out['report']

        # self.assertTrue(report.converged())
        self.assertTrue(confs[0].is_file())
        self.assertTrue(confs[1].is_file())
        self.assertTrue(confs[0].read_text(), 'conf of conf.0')
        self.assertTrue(confs[1].read_text(), 'conf of conf.1')
        

class TestSelectConfs(unittest.TestCase):
    def setUp(self): 
        self.conf_selector = MockedConfSelector()
        self.traj_fmt = 'foo'
        self.type_map = []
        self.trajs = [Path('traj.foo'), Path('traj.bar')]
        self.model_devis = [Path('md.foo'), Path('md.bar')]

    def tearDown(self):
        for ii in ['conf.0', 'conf.1']:
            ii=Path(ii)
            if ii.is_file():
                os.remove(ii)
    
    def test(self):
        op = SelectConfs()
        out = op.execute(OPIO({
            'conf_selector': self.conf_selector,
            'traj_fmt': self.traj_fmt,
            'type_map' : self.type_map,
            'trajs' : self.trajs,
            'model_devis' : self.model_devis,
        }))
        confs = out['confs']
        report = out['report']

        # self.assertTrue(report.converged())
        self.assertTrue(confs[0].is_file())
        self.assertTrue(confs[1].is_file())
        self.assertTrue(confs[0].read_text(), 'conf of conf.0')
        self.assertTrue(confs[1].read_text(), 'conf of conf.1')
        
        
        
