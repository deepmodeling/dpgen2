import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dpgen2
upload_python_package = '../dpgen2'
if os.getenv('SKIP_UT_WITH_DFLOW'):
    skip_ut_with_dflow = (int(os.getenv('SKIP_UT_WITH_DFLOW')) != 0)
    skip_ut_with_dflow_reason = 'skip because environment variable SKIP_UT_WITH_DFLOW is set to non-zero'
else:
    skip_ut_with_dflow = False
    skip_ut_with_dflow_reason = ''
