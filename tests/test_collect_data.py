import shutil
import time
import unittest
from pathlib import (
    Path,
)

from dflow import (
    Step,
    Workflow,
    download_artifact,
    upload_artifact,
)
from dflow.python import (
    OPIO,
    PythonOPTemplate,
)

try:
    from context import (
        dpgen2,
    )
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from context import (
    default_host,
    default_image,
    skip_ut_with_dflow,
    skip_ut_with_dflow_reason,
    upload_python_packages,
)
from mocked_ops import (
    MockedCollectData,
)


class TestMockedCollectData(unittest.TestCase):
    def setUp(self):
        self.iter_data = ["foo/iter0", "bar/iter1"]
        self.iter_data = [Path(ii) for ii in self.iter_data]
        self.name = "outdata"
        self.labeled_data = ["d0", "d1"]
        self.labeled_data = [Path(ii) for ii in self.labeled_data]
        for ii in self.iter_data:
            ii.mkdir(exist_ok=True, parents=True)
            (ii / "data").write_text(f"data of {ii!s}")
        for ii in self.labeled_data:
            (ii).mkdir(exist_ok=True, parents=True)
            (ii / "data").write_text(f"data of {ii!s}")
        self.type_map = []

    def tearDown(self):
        for ii in ["d0", "d1", "outdata", "foo", "bar", "iter0", "iter1"]:
            ii = Path(ii)
            if ii.is_dir():
                shutil.rmtree(ii)

    def test(self):
        op = MockedCollectData()
        out = op.execute(
            OPIO(
                {
                    "name": self.name,
                    "labeled_data": self.labeled_data,
                    "iter_data": self.iter_data,
                    "type_map": self.type_map,
                }
            )
        )
        iter_data = out["iter_data"]

        out_data = Path(self.name)
        self.assertTrue(out_data.is_dir())
        self.assertTrue((out_data / "d0").is_dir())
        self.assertTrue((out_data / "d1").is_dir())
        self.assertTrue((out_data / "d0" / "data").read_text(), "data of d0")
        self.assertTrue((out_data / "d1" / "data").read_text(), "data of d1")
        path = Path("iter0")
        self.assertTrue(path.is_dir())
        self.assertTrue((path / "data").read_text(), "data of iter0")
        path = Path("iter1")
        self.assertTrue(path.is_dir())
        self.assertTrue((path / "data").read_text(), "data of iter1")


@unittest.skipIf(skip_ut_with_dflow, skip_ut_with_dflow_reason)
class TestMockedCollectDataArgo(unittest.TestCase):
    def setUp(self):
        self.iter_data = set(("foo/iter0", "bar/iter1"))
        self.iter_data = set([Path(ii) for ii in self.iter_data])
        self.name = "outdata"
        self.labeled_data = ["d0", "d1"]
        self.labeled_data = [Path(ii) for ii in self.labeled_data]
        for ii in self.iter_data:
            ii.mkdir(exist_ok=True, parents=True)
            (ii / "data").write_text(f"data of {ii!s}")
        for ii in self.labeled_data:
            (ii).mkdir(exist_ok=True, parents=True)
            (ii / "data").write_text(f"data of {ii!s}")
        self.iter_data = upload_artifact(list(self.iter_data))
        self.labeled_data = upload_artifact(self.labeled_data)
        self.type_map = []

    def tearDown(self):
        for ii in ["d0", "d1", "outdata", "foo", "bar", "iter0", "iter1"]:
            ii = Path(ii)
            if ii.is_dir():
                shutil.rmtree(ii)

    def test(self):
        coll_data = Step(
            "coll-data",
            template=PythonOPTemplate(
                MockedCollectData,
                image=default_image,
                output_artifact_archive={
                    "iter_data": None,
                },
                python_packages=upload_python_packages,
            ),
            parameters={
                "name": self.name,
                "type_map": self.type_map,
            },
            artifacts={
                "iter_data": self.iter_data,
                "labeled_data": self.labeled_data,
            },
        )

        wf = Workflow(name="coll", host=default_host)
        wf.add(coll_data)
        wf.submit()

        while wf.query_status() in ["Pending", "Running"]:
            time.sleep(2)

        self.assertEqual(wf.query_status(), "Succeeded")
        step = wf.query_step(name="coll-data")[0]
        self.assertEqual(step.phase, "Succeeded")

        download_artifact(step.outputs.artifacts["iter_data"])

        out_data = Path(self.name)
        self.assertTrue(out_data.is_dir())
        self.assertTrue((out_data / "d0").is_dir())
        self.assertTrue((out_data / "d1").is_dir())
        self.assertTrue((out_data / "d0" / "data").read_text(), "data of d0")
        self.assertTrue((out_data / "d1" / "data").read_text(), "data of d1")
        path = Path("iter0")
        self.assertTrue(path.is_dir())
        self.assertTrue((path / "data").read_text(), "data of iter0")
        path = Path("iter1")
        self.assertTrue(path.is_dir())
        self.assertTrue((path / "data").read_text(), "data of iter1")
