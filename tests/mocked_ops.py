from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    upload_packages,
    FatalError,
)

upload_packages.append(__file__)

import os, json, shutil, re, pickle
from pathlib import Path
from typing import Tuple, List
try:
    from flow.context import dpgen2
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from dpgen2.constants import (
    train_task_pattern,
    train_script_name,
    train_log_name,
    model_name_pattern,
    lmp_task_pattern,
    lmp_conf_name,
    lmp_input_name,
    lmp_traj_name,
    lmp_log_name,
    lmp_model_devi_name,
    vasp_task_pattern,
    vasp_conf_name,
    vasp_input_name,
)
from dpgen2.op.run_dp_train import RunDPTrain
from dpgen2.op.prep_dp_train import PrepDPTrain
from dpgen2.op.prep_lmp import PrepExplorationTaskGroup
from dpgen2.op.run_lmp import RunLmp
from dpgen2.op.prep_vasp import PrepVasp
from dpgen2.op.run_vasp import RunVasp
from dpgen2.op.collect_data import CollectData
from dpgen2.op.select_confs import SelectConfs
from dpgen2.exploration.selector import TrustLevel, ConfSelector
from dpgen2.exploration.task import ExplorationTask, ExplorationTaskGroup, ExplorationStage
from dpgen2.exploration.report import ExplorationReport
from dpgen2.exploration.scheduler import ConvergenceCheckStageScheduler
from dpgen2.fp.vasp import VaspInputs
from dpgen2.utils import (
    load_object_from_file,
    dump_object_to_file,
)

mocked_template_script = { 'seed' : 1024, 'data': [] }
mocked_numb_models = 3
mocked_numb_lmp_tasks = 6
mocked_numb_select = 2
mocked_incar_template = 'incar template'

def make_mocked_init_models(numb_models):
    tmp_models = []
    for ii in range(numb_models):
        ff = Path(model_name_pattern % ii)
        ff.write_text(f'This is init model {ii}')
        tmp_models.append(ff)
    return tmp_models

def make_mocked_init_data():
    tmp_init_data = [Path('init_data/foo'), Path('init_data/bar')]
    for ii in tmp_init_data:
        ii.mkdir(exist_ok=True, parents=True)
        (ii/'a').write_text('data a')
        (ii/'b').write_text('data b')
    return tmp_init_data



class MockedPrepDPTrain(PrepDPTrain):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        template = ip['template_script']
        numb_models = ip['numb_models']
        ofiles = []
        osubdirs = []
        
        assert(template == mocked_template_script)
        assert(numb_models == mocked_numb_models)

        for ii in range(numb_models):
            jtmp = template
            jtmp['seed'] = ii
            subdir = Path(train_task_pattern % ii) 
            subdir.mkdir(exist_ok=True, parents=True)
            fname = subdir / 'input.json'
            with open(fname, 'w') as fp:
                json.dump(jtmp, fp, indent = 4)
            osubdirs.append(str(subdir))
            ofiles.append(fname)

        op = OPIO({
            "task_names" : osubdirs,
            "task_paths" : [Path(ii) for ii in osubdirs],
        })
        return op


class MockedRunDPTrain(RunDPTrain):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        work_dir = Path(ip['task_name'])
        script = ip['task_path'] / 'input.json'
        init_model = Path(ip['init_model'])
        init_data = ip['init_data']
        iter_data = ip['iter_data']

        assert(script.is_file())
        assert(ip['task_path'].is_dir())
        assert(init_model.is_file())
        assert(len(init_data) == 2)
        assert(re.match('task.[0-9][0-9][0-9][0-9]', ip['task_name']))
        task_id = int(ip['task_name'].split('.')[1])
        assert(ip['task_name'] in str(ip['task_path']))
        assert("model" in str(ip['init_model']))
        assert(".pb" in str(ip['init_model']))
        list_init_data = sorted([str(ii) for ii in init_data] )
        assert('init_data/bar' in list_init_data[0])
        assert('init_data/foo' in list_init_data[1])        
        assert(Path(list_init_data[0]).is_dir())
        assert(Path(list_init_data[1]).is_dir())

        script = Path(script).resolve()
        init_model = init_model.resolve()
        init_model_str = str(init_model)
        init_data = [ii.resolve() for ii in init_data]
        iter_data = [ii.resolve() for ii in iter_data]
        init_data_str = [str(ii) for ii in init_data]
        iter_data_str = [str(ii) for ii in iter_data]

        with open(script) as fp:
            jtmp = json.load(fp)        
        data = []
        for ii in sorted(init_data_str):
            data.append(ii)
        for ii in sorted(iter_data_str):
            data.append(ii)
        jtmp['data'] = data
        with open(script, 'w') as fp:
            json.dump(jtmp, fp, indent=4)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        oscript = Path('input.json')
        if not oscript.exists():
            from shutil import copyfile
            copyfile(script, oscript)
        model = Path('model.pb')
        lcurve = Path('lcurve.out')
        log = Path('log')

        assert(init_model.exists())        
        with log.open("w") as f:
            f.write(f'init_model {str(init_model)} OK\n')
        for ii in jtmp['data']:
            assert(Path(ii).exists())
            assert((ii in init_data_str) or (ii in iter_data_str))
            with log.open("a") as f:
                f.write(f'data {str(ii)} OK\n')
        assert(script.exists())
        with log.open("a") as f:
            f.write(f'script {str(script)} OK\n')

        with model.open("w") as f:
            f.write('read from init model: \n')
            f.write(init_model.read_text() + '\n')
        with lcurve.open("w") as f:
            f.write('read from train_script: \n')
            f.write(script.read_text() + '\n')

        os.chdir(cwd)
        
        return OPIO({
            'script' : work_dir/oscript,
            'model' : work_dir/model,
            'lcurve' : work_dir/lcurve,
            'log' : work_dir/log
        })


class MockedRunDPTrainNoneInitModel(RunDPTrain):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        work_dir = Path(ip['task_name'])
        script = ip['task_path'] / 'input.json'
        if ip['init_model'] is not None:
            raise FatalError('init model is not None')
        init_data = ip['init_data']
        iter_data = ip['iter_data']

        assert(script.is_file())
        assert(ip['task_path'].is_dir())
        assert(len(init_data) == 2)
        assert(re.match('task.[0-9][0-9][0-9][0-9]', ip['task_name']))
        task_id = int(ip['task_name'].split('.')[1])
        assert(ip['task_name'] in str(ip['task_path']))
        list_init_data = sorted([str(ii) for ii in init_data] )
        assert('init_data/bar' in list_init_data[0])
        assert('init_data/foo' in list_init_data[1])        
        assert(Path(list_init_data[0]).is_dir())
        assert(Path(list_init_data[1]).is_dir())

        script = Path(script).resolve()
        init_data = [ii.resolve() for ii in init_data]
        iter_data = [ii.resolve() for ii in iter_data]
        init_data_str = [str(ii) for ii in init_data]
        iter_data_str = [str(ii) for ii in iter_data]

        with open(script) as fp:
            jtmp = json.load(fp)        
        data = []
        for ii in sorted(init_data_str):
            data.append(ii)
        for ii in sorted(iter_data_str):
            data.append(ii)
        jtmp['data'] = data
        with open(script, 'w') as fp:
            json.dump(jtmp, fp, indent=4)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        oscript = Path('input.json')
        if not oscript.exists():
            from shutil import copyfile
            copyfile(script, oscript)
        model = Path('model.pb')
        lcurve = Path('lcurve.out')
        log = Path('log')

        for ii in jtmp['data']:
            assert(Path(ii).exists())
            assert((ii in init_data_str) or (ii in iter_data_str))
            with log.open("a") as f:
                f.write(f'data {str(ii)} OK\n')
        assert(script.exists())
        with log.open("a") as f:
            f.write(f'script {str(script)} OK\n')

        with model.open("w") as f:
            f.write('read from init model: \n')
        with lcurve.open("w") as f:
            f.write('read from train_script: \n')
            f.write(script.read_text() + '\n')

        os.chdir(cwd)
        
        return OPIO({
            'script' : work_dir/oscript,
            'model' : work_dir/model,
            'lcurve' : work_dir/lcurve,
            'log' : work_dir/log
        })


class MockedRunLmp(RunLmp):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        task_name = ip['task_name']
        task_path = ip['task_path']
        models = ip['models']

        assert(ip['task_path'].is_dir())
        assert(re.match('task.[0-9][0-9][0-9][0-9][0-9][0-9]', ip['task_name']))
        task_id = int(ip['task_name'].split('.')[1])
        assert(task_path.is_dir())
        assert(ip['task_name'] in str(ip['task_path']))
        assert(len(models) == mocked_numb_models)
        for ii in range(mocked_numb_models):
            assert(ip['models'][ii].is_file())
            assert("model" in str(ip['models'][ii]))
            assert(".pb" in str(ip['models'][ii]))
        assert((task_path/lmp_conf_name).is_file())
        assert((task_path/lmp_input_name).is_file())
        
        task_path = task_path.resolve()
        models = [ii.resolve() for ii in models]
        models_str = [str(ii) for ii in models]
        
        work_dir = Path(task_name)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        import glob
        ifiles = glob.glob(str(task_path / '*'))
        for ii in ifiles:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)
        for ii in models:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)
        
        log = Path(lmp_log_name)
        traj = Path(lmp_traj_name)
        model_devi = Path(lmp_model_devi_name)
        
        # fc = ['log of {task_name}']
        # for ii in ['conf.lmp', 'in.lammps'] + models_str:
        #     if Path(ii).exists():
        #         fc.append(f'{ii} OK')
        # log.write_text('\n'.join(fc))        
        # log.write_text('log of {task_name}')
        fc = []
        for ii in [lmp_conf_name, lmp_input_name] + [ii.name for ii in models]:
             fc.append(Path(ii).read_text())
        log.write_text('\n'.join(fc))
        model_devi.write_text(f'model_devi of {task_name}')
        traj_out = []
        traj_out.append(f'traj of {task_name}')
        traj_out.append(Path(lmp_conf_name).read_text())
        traj_out.append(Path(lmp_input_name).read_text())
        traj.write_text('\n'.join(traj_out))

        os.chdir(cwd)

        return OPIO({
            'log' : work_dir/log,
            'traj' : work_dir/traj,
            'model_devi' : work_dir/model_devi,
        })


class MockedPrepVasp(PrepVasp):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        confs = ip['confs']
        # incar_temp = ip['incar_temp']
        # potcars = ip['potcars']
        vasp_input_fname = ip['inputs']
        vasp_input = load_object_from_file(vasp_input_fname)
        type_map = ip['type_map']
        if not (type_map == ['H', 'O']):
            raise FatalError

        incar_temp = vasp_input.incar_template
        potcars = vasp_input.potcars

        for ii in confs:
            assert(ii.is_file())
        assert(vasp_input.incar_template == mocked_incar_template)

        nconfs = len(confs)
        task_paths = []

        for ii in range(nconfs):
            task_path = Path(vasp_task_pattern % ii)
            task_path.mkdir(exist_ok=True, parents=True)
            from shutil import copyfile
            copyfile(confs[ii], task_path/vasp_conf_name)
            (task_path/vasp_input_name).write_text(incar_temp)
            task_paths.append(task_path)

        task_names = [str(ii) for ii in task_paths]
        return OPIO({
            'task_names' : task_names,
            'task_paths' : task_paths,
        })


class MockedRunVasp(RunVasp):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        task_name = ip['task_name']
        task_path = ip['task_path']

        assert(ip['task_path'].is_dir())
        assert(re.match('task.[0-9][0-9][0-9][0-9][0-9][0-9]', ip['task_name']))
        task_id = int(ip['task_name'].split('.')[1])
        assert(ip['task_name'] in str(ip['task_path']))
        assert((ip['task_path']/vasp_conf_name).is_file())
        assert((ip['task_path']/vasp_input_name).is_file())

        work_dir = Path(task_name)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        import glob
        ifiles = glob.glob(str(task_path / '*'))
        for ii in ifiles:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)
        
        log = Path('log')
        # labeled_data = Path('labeled_data')
        labeled_data = Path('data_'+task_name)
        
        fc = []
        for ii in [vasp_conf_name, vasp_input_name]:
             fc.append(Path(ii).read_text())
        log.write_text('\n'.join(fc))
        labeled_data.mkdir(exist_ok=True, parents=True)

        fc = []
        fc.append(f'labeled_data of {task_name}')
        fc.append(Path(vasp_conf_name).read_text())
        (labeled_data / 'data').write_text('\n'.join(fc))

        os.chdir(cwd)

        return OPIO({
            'log' : work_dir/log,
            'labeled_data' : work_dir/labeled_data,
        })


class MockedRunVaspFail1(RunVasp):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        task_name = ip['task_name']
        task_path = ip['task_path']

        assert(ip['task_path'].is_dir())
        assert(re.match('task.[0-9][0-9][0-9][0-9][0-9][0-9]', ip['task_name']))
        task_id = int(ip['task_name'].split('.')[1])
        assert(ip['task_name'] in str(ip['task_path']))
        assert((ip['task_path']/vasp_conf_name).is_file())
        assert((ip['task_path']/vasp_input_name).is_file())

        if task_id == 1:
            raise FatalError

        work_dir = Path(task_name)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        import glob
        ifiles = glob.glob(str(task_path / '*'))
        for ii in ifiles:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)
        
        log = Path('log')
        # labeled_data = Path('labeled_data')
        labeled_data = Path('data_'+task_name)
        
        fc = []
        for ii in [vasp_conf_name, vasp_input_name]:
             fc.append(Path(ii).read_text())
        log.write_text('\n'.join(fc))
        labeled_data.mkdir(exist_ok=True, parents=True)

        fc = []
        fc.append(f'labeled_data of {task_name}')
        fc.append(Path(vasp_conf_name).read_text())
        (labeled_data / 'data').write_text('\n'.join(fc))

        os.chdir(cwd)

        return OPIO({
            'log' : work_dir/log,
            'labeled_data' : work_dir/labeled_data,
        })


class MockedRunVaspRestart(RunVasp):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        task_name = ip['task_name']
        task_path = ip['task_path']

        assert(ip['task_path'].is_dir())
        assert(re.match('task.[0-9][0-9][0-9][0-9][0-9][0-9]', ip['task_name']))
        task_id = int(ip['task_name'].split('.')[1])
        assert(ip['task_name'] in str(ip['task_path']))
        assert((ip['task_path']/vasp_conf_name).is_file())
        assert((ip['task_path']/vasp_input_name).is_file())

        work_dir = Path(task_name)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        import glob
        ifiles = glob.glob(str(task_path / '*'))
        for ii in ifiles:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)
        
        log = Path('log')
        # labeled_data = Path('labeled_data')
        labeled_data = Path('data_'+task_name)
        
        fc = []
        for ii in [vasp_conf_name, vasp_input_name]:
             fc.append(Path(ii).read_text())
        log.write_text('\n'.join(fc))
        labeled_data.mkdir(exist_ok=True, parents=True)

        fc = []
        fc.append('restarted')
        fc.append(f'labeled_data of {task_name}')
        fc.append(Path(vasp_conf_name).read_text())
        (labeled_data / 'data').write_text('\n'.join(fc))

        os.chdir(cwd)

        return OPIO({
            'log' : work_dir/log,
            'labeled_data' : work_dir/labeled_data,
        })


class MockedCollectData(CollectData):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        name = ip['name']
        labeled_data = ip['labeled_data']
        iter_data = ip['iter_data']

        new_iter_data = []
        # copy iter_data
        for ii in iter_data:
            iiname = ii.name
            shutil.copytree(ii, iiname)
            new_iter_data.append(Path(iiname))

        # collect labled data
        name = Path(name)
        name.mkdir(exist_ok=True, parents=True)
        
        for ii in labeled_data:
            iiname = ii.name
            shutil.copytree(ii, name/iiname)
        new_iter_data.append(name)
        
        return OPIO({
            "iter_data" : new_iter_data,
        })


class MockedCollectDataFailed(CollectData):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        name = ip['name']
        labeled_data = ip['labeled_data']
        name = Path(name)
        for ii in labeled_data:
            iiname = ii.name
            print('copy-------------------', ii, name/iiname)

        raise FatalError


class MockedCollectDataRestart(CollectData):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        name = ip['name']
        labeled_data = ip['labeled_data']
        iter_data = ip['iter_data']

        new_iter_data = []
        # copy iter_data
        for ii in iter_data:
            iiname = ii.name
            shutil.copytree(ii, iiname)
            new_iter_data.append(Path(iiname))

        # collect labled data
        name = Path(name)
        name.mkdir(exist_ok=True, parents=True)
        
        for ii in labeled_data:
            iiname = ii.name
            print('copy-------------------', ii, name/iiname)
            shutil.copytree(ii, name/iiname)
            fc = (name/iiname/'data').read_text()
            fc = "restart\n" + fc
            (name/iiname/'data').write_text(fc)

        new_iter_data.append(name)
        
        return OPIO({
            "iter_data" : new_iter_data,
        })


class MockedExplorationReport(ExplorationReport):
    def __init__(self):
        self.failed = .1
        self.candidate = .1
        self.accurate = .8

    def failed_ratio (
            self, 
            tag = None,
    ) -> float :
        return self.failed

    def accurate_ratio (
            self,
            tag = None,
    ) -> float :
        return self.accurate

    def candidate_ratio (
            self,
            tag = None,
    ) -> float :
        return self.candidate


class MockedExplorationTaskGroup(ExplorationTaskGroup):
    def __init__(self):
        super().__init__()
        ntask = mocked_numb_lmp_tasks
        for jj in range(ntask):
            tt = ExplorationTask()
            tt\
                .add_file(lmp_conf_name, f'mocked conf {jj}')\
                .add_file(lmp_input_name, f'mocked input {jj}')
            self.add_task(tt)

class MockedExplorationTaskGroup1(ExplorationTaskGroup):
    def __init__(self):
        super().__init__()
        ntask = mocked_numb_lmp_tasks
        for jj in range(ntask):
            tt = ExplorationTask()
            tt\
                .add_file(lmp_conf_name, f'mocked 1 conf {jj}')\
                .add_file(lmp_input_name, f'mocked 1 input {jj}')
            self.add_task(tt)

class MockedExplorationTaskGroup2(ExplorationTaskGroup):
    def __init__(self):
        super().__init__()
        ntask = mocked_numb_lmp_tasks
        for jj in range(ntask):
            tt = ExplorationTask()
            tt\
                .add_file(lmp_conf_name, f'mocked 2 conf {jj}')\
                .add_file(lmp_input_name, f'mocked 2 input {jj}')
            self.add_task(tt)

class MockedStage(ExplorationStage):
    def make_task(self):
        return MockedExplorationTaskGroup()

class MockedStage1(ExplorationStage):
    def make_task(self):
        return MockedExplorationTaskGroup1()

class MockedStage2(ExplorationStage):
    def make_task(self):
        return MockedExplorationTaskGroup2()

class MockedConfSelector(ConfSelector):
    def __init__(
            self,
            trust_level: TrustLevel = TrustLevel(0.1, 0.2),
    ):
        self.trust_level = trust_level

    def select (
            self,
            trajs : List[Path],
            model_devis : List[Path],
            traj_fmt : str = 'deepmd/npy',
            type_map : List[str] = None,
    ) -> Tuple[List[ Path ], TrustLevel] :
        confs = []
        if len(trajs) == mocked_numb_lmp_tasks:
            # get output from prep_run_lmp
            for ii in range(mocked_numb_select):
                ftraj = trajs[ii].read_text().strip().split('\n')
                fcont = []
                fcont.append(f'select conf.{ii}')
                fcont.append(ftraj[1])
                fcont.append(ftraj[2])
                fname = Path(f'conf.{ii}')
                fname.write_text('\n'.join(fcont))
                confs.append(fname)
        else:
            # for the case of artificial input. trajs of length 2
            fname = Path('conf.0')
            fname.write_text('conf of conf.0')
            confs.append(fname)
            fname = Path('conf.1')
            fname.write_text('conf of conf.1')
            confs.append(fname)
        report = MockedExplorationReport()
        return confs, report

class MockedSelectConfs(SelectConfs):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        conf_selector = ip['conf_selector']
        trajs = ip['trajs']
        model_devis = ip['model_devis']
        confs, report = conf_selector.select(trajs, model_devis)

        # get lmp output. check if all trajs and model devis are files
        if len(trajs) == mocked_numb_lmp_tasks:
            for ii in range(mocked_numb_lmp_tasks):
                assert(trajs[ii].is_file())
                assert(model_devis[ii].is_file())

        return OPIO({
            "report" : report,
            "confs" : confs,
        })


class MockedConstTrustLevelStageScheduler(ConvergenceCheckStageScheduler):
    def __init__(
            self,
            stage : ExplorationStage,
            trust_level : TrustLevel,
            conv_accuracy : float = 0.9,
            max_numb_iter : int = None,
    ):
        self.selector = MockedConfSelector(trust_level)
        super().__init__(stage, self.selector, conv_accuracy, max_numb_iter)
