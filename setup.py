# -*- coding: utf-8 -*-

from os import path
from pathlib import Path
import setuptools

# define constants
INSTALL_REQUIRES = (Path(__file__).parent / "requirements.txt").read_text().splitlines()
setup_requires = ["setuptools_scm"]

readme_file = Path(__file__).parent / "README.md"
readme = readme_file.read_text(encoding="utf-8")

setuptools.setup(
    name="dpgen2",
    use_scm_version={'write_to': 'dpgen2/_version.py'},
    setup_requires=['setuptools_scm'],
    author="Han Wang",
    author_email="wang_han@iapcm.ac.cn",
    description="DPGEN2: concurrent learning workflow generating the machine learning potential energy models.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/deepmodeling/dpgen2",
    packages=[
        "dpgen2",
        "dpgen2/entrypoint",
        "dpgen2/op",
        "dpgen2/superop",
        "dpgen2/flow",
        "dpgen2/fp",
        "dpgen2/utils",
        "dpgen2/exploration",
        "dpgen2/exploration/report",
        "dpgen2/exploration/scheduler",
        "dpgen2/exploration/task",
        "dpgen2/exploration/task/lmp",
        "dpgen2/exploration/selector",
    ],
    package_data={'dpgen2':['*.json']},
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    ],
    keywords='deep potential concurrent learning',
    install_requires=INSTALL_REQUIRES,
    entry_points={
        "console_scripts": [
            "dpgen2 = dpgen2.entrypoint.main:main"
        ]
    },
    extras_require={
        'docs': ['sphinx', 'recommonmark', 'sphinx_rtd_theme>=1.0.0rc1', 'numpydoc', 'm2r2'],
    }
)

