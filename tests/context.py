import sys, os

dpgen_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, dpgen_path)
import dpgen2
from dpgen2.utils import dflow_config

if os.getenv("SKIP_UT_WITH_DFLOW"):
    skip_ut_with_dflow = int(os.getenv("SKIP_UT_WITH_DFLOW")) != 0
    skip_ut_with_dflow_reason = (
        "skip because environment variable SKIP_UT_WITH_DFLOW is set to non-zero"
    )
else:
    skip_ut_with_dflow = False
    skip_ut_with_dflow_reason = ""
upload_python_packages = [os.path.join(dpgen_path, "dpgen2")]
# one needs to set proper values for the following variable.
default_image = "dptechnology/dpgen2:latest"
default_host = None
dflow_config({})
if os.getenv("DFLOW_DEBUG"):
    from dflow import config

    config["mode"] = "debug"
