from .context import dpgen2
import numpy as np
import unittest, json, shutil, os
import random
import tempfile
import dpdata
from pathlib import Path
from dpgen2.entrypoint.submit import (
    expand_idx,
    print_list_steps,
    update_reuse_step_scheduler,
)

ifc0 = """Al1 
1.0
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 2.0
Al 
1 
cartesian
   0.0000000000    0.0000000000    0.0000000000
"""
ofc0 = '\n1 atoms\n2 atom types\n   0.0000000000    2.0000000000 xlo xhi\n   0.0000000000    2.0000000000 ylo yhi\n   0.0000000000    2.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\nAtoms # atomic\n\n     1      1    0.0000000000    0.0000000000    0.0000000000\n'

ifc1 = """Mg1 
1.0
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
Mg 
1 
cartesian
   0.0000000000    0.0000000000    0.0000000000
"""
ofc1 = '\n1 atoms\n2 atom types\n   0.0000000000    3.0000000000 xlo xhi\n   0.0000000000    3.0000000000 ylo yhi\n   0.0000000000    3.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\nAtoms # atomic\n\n     1      2    0.0000000000    0.0000000000    0.0000000000\n'

ifc2 = """Mg1 
1.0
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
Mg 
1 
cartesian
   0.0000000000    0.0000000000    0.0000000000
"""
ofc2 = '\n1 atoms\n2 atom types\n   0.0000000000    4.0000000000 xlo xhi\n   0.0000000000    4.0000000000 ylo yhi\n   0.0000000000    4.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\nAtoms # atomic\n\n     1      2    0.0000000000    0.0000000000    0.0000000000\n'


class MockedScheduler():
    def __init__(self, value=0):
        self.value = value

class MockedStep():
    def __init__(self, scheduler=None):
        self.scheduler = scheduler
        self.key = f"iter-{self.scheduler.value}--scheduler"

    def modify_output_artifact(self, key, scheduler):
        assert key == "exploration_scheduler"
        self.scheduler = scheduler


class TestSubmit(unittest.TestCase):
    def test_expand_idx(self):
        ilist = ['1', '3-5', '10-20:2']
        olist = expand_idx(ilist)
        expected_olist = [1, 3, 4, 10, 12, 14, 16, 18]
        self.assertEqual(olist, expected_olist)


    def test_print_list_steps(self):
        ilist = ['foo', 'bar']
        ostr = print_list_steps(ilist)
        expected_ostr = '       0    foo\n       1    bar'
        self.assertEqual(ostr, expected_ostr)


    def test_update_reuse_step_scheduler(self):
        reuse_steps = [
            MockedStep(MockedScheduler(0)),
            MockedStep(MockedScheduler(1)),
            MockedStep(MockedScheduler(2)),
            MockedStep(MockedScheduler(3)),
        ]
        reuse_steps = update_reuse_step_scheduler(
            reuse_steps, 
            MockedScheduler(4),
        )
        self.assertEqual(len(reuse_steps), 4)
        self.assertEqual(reuse_steps[0].scheduler.value, 0)
        self.assertEqual(reuse_steps[1].scheduler.value, 1)
        self.assertEqual(reuse_steps[2].scheduler.value, 2)
        self.assertEqual(reuse_steps[3].scheduler.value, 4)
