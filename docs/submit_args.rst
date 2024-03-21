.. _submitargs:

Arguments of the submit script
==============================
.. note::
   One can load, modify, and export the input file by using our effective web-based tool `DP-GUI <https://deepmodeling.com/dpgui/input/dpgen2-submit>`_ online or hosted using the :ref:`command line interface <cli>` :code:`dpgen2 gui`. All parameters below can be set in DP-GUI. By clicking "SAVE JSON", one can download the input file.

.. dargs::
   :module: dpgen2.entrypoint.args
   :func: submit_args


.. _task_group_sec:

Task group definition
---------------------

LAMMPS task group
^^^^^^^^^^^^^^^^^

.. dargs::
   :module: dpgen2.exploration.task
   :func: lmp_task_group_args

CALYPSO task group
^^^^^^^^^^^^^^^^^^

.. dargs::
   :module: dpgen2.exploration.task
   :func: caly_task_group_args
