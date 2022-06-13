import os, json, dpdata, glob
from pathlib import Path
from dpgen2.utils.run_command import run_command
from dpgen2.utils.chdir import set_directory
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    TransientError,
    FatalError,
)
from typing import (
    Tuple, 
    List, 
)
from dpgen2.constants import (
    train_task_pattern,
    train_script_name,
)
from dargs import (
    dargs, 
    Argument, 
    Variant, 
    ArgumentEncoder,
)


class RunDPTrain(OP):
    r"""Execute a DP training task. Train and freeze a DP model. 

    A working directory named `task_name` is created. All input files
    are copied or symbol linked to directory `task_name`. The
    DeePMD-kit training and freezing commands are exectuted from
    directory `task_name`.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "config" : dict,
            "task_name" : str,
            "task_path" : Artifact(Path),
            "init_model" : Artifact(Path, optional=True),
            "init_data" : Artifact(List[Path]),
            "iter_data" : Artifact(List[Path]),
        })
    
    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "script" : Artifact(Path),
            "model" : Artifact(Path),
            "lcurve" : Artifact(Path),
            "log" : Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:
        
            - `config`: (`dict`) The config of training task. Check `RunDPTrain.training_args` for definitions.
            - `task_name`: (`str`) The name of training task.
            - `task_path`: (`Artifact(Path)`) The path that contains all input files prepareed by `PrepDPTrain`.
            - `init_model`: (`Artifact(Path)`) A frozen model to initialize the training.
            - `init_data`: (`Artifact(List[Path])`) Initial training data.
            - `iter_data`: (`Artifact(List[Path])`) Training data generated in the DPGEN iterations.

        Returns
        -------
            Output dict with components:
        
            - `script`: (`Artifact(Path)`) The training script.
            - `model`: (`Artifact(Path)`) The trained frozen model.
            - `lcurve`: (`Artifact(Path)`) The learning curve file.
            - `log`: (`Artifact(Path)`) The log file of training.
        
        Exceptions
        ----------
        FatalError
            On the failure of training or freezing. Human intervention needed.
        """
        config = ip['config'] if ip['config'] is not None else {}
        config = RunDPTrain.normalize_config(config)
        task_name = ip['task_name']
        task_path = ip['task_path']
        init_model = ip['init_model']
        init_data = ip['init_data']
        iter_data = ip['iter_data']
        iter_data_old_exp = _expand_all_multi_sys_to_sys(iter_data[:-1])
        iter_data_new_exp = _expand_all_multi_sys_to_sys(iter_data[-1:])
        iter_data_exp = iter_data_old_exp + iter_data_new_exp
        work_dir = Path(task_name)

        # update the input script
        input_script = Path(task_path)/train_script_name
        with open(input_script) as fp:
            train_dict = json.load(fp)
        if "systems" in train_dict['training']:
            major_version = "1"
        else:
            major_version = "2"

        # auto prob style
        do_init_model = RunDPTrain.decide_init_model(config, init_model, init_data, iter_data)
        auto_prob_str = "prob_sys_size"
        if do_init_model:
            old_ratio = config['init_model_old_ratio']
            numb_old = len(init_data) + len(iter_data_old_exp)
            numb_new = numb_old + len(iter_data_new_exp)
            auto_prob_str = f"prob_sys_size; 0:{numb_old}:{old_ratio}; {numb_old}:{numb_new}:{1.-old_ratio:g}"

        # update the input dict
        train_dict = RunDPTrain.write_data_to_input_script(
            train_dict, init_data, iter_data_exp, auto_prob_str, major_version)
        train_dict = RunDPTrain.write_other_to_input_script(
            train_dict, config, do_init_model, major_version)        

        with set_directory(work_dir):
            # open log
            fplog = open('train.log', 'w')
            def clean_before_quit():
                fplog.close()

            # dump train script
            with open(train_script_name, 'w') as fp:
                json.dump(train_dict, fp, indent=4)

            # train model
            if do_init_model:
                command = ['dp', 'train', '--init-frz-model', str(init_model), train_script_name]
            else:
                command = ['dp', 'train', train_script_name]
            ret, out, err = run_command(command)
            if ret != 0:
                clean_before_quit()
                raise FatalError(
                    'dp train failed\n',
                    'out msg', out, '\n',
                    'err msg', err, '\n'
                )
            fplog.write(out)

            # freeze model
            ret, out, err = run_command(['dp', 'freeze', '-o', 'frozen_model.pb'])
            if ret != 0:
                clean_before_quit()
                raise FatalError(
                    'dp freeze failed\n',
                    'out msg', out, '\n',
                    'err msg', err, '\n'
                )
            fplog.write(out)

            clean_before_quit()
        
        return OPIO({
            "script" : work_dir / train_script_name,
            "model" : work_dir / "frozen_model.pb",
            "lcurve" : work_dir / "lcurve.out",
            "log" : work_dir / "train.log",
        })
            

    @staticmethod
    def write_data_to_input_script(
            idict : dict,
            init_data : List[Path],
            iter_data : List[Path],
            auto_prob_str : str = "prob_sys_size",
            major_version : str = "1",
    ):
        odict = idict.copy()
        data_list = [str(ii) for ii in init_data] + [str(ii) for ii in iter_data]
        if major_version == "1":
            # v1 behavior
            odict['training']['systems'] = data_list
            odict['training']['batch_size'] = "auto"
            odict['training']['auto_prob_style'] = auto_prob_str
        elif major_version == "2":
            # v2 behavior
            odict['training']['training_data']['systems'] = data_list
            odict['training']['training_data']['batch_size'] = "auto"
            odict['training']['training_data']['auto_prob'] = auto_prob_str
            odict['training'].pop('validation_data', None)
        else:
            raise RuntimeError('unsupported DeePMD-kit major version', major_version)
        return odict

    @staticmethod
    def write_other_to_input_script(
            idict,
            config,
            do_init_model,
            major_version : str = "1",
    ):
        odict = idict.copy()
        odict['training']['disp_file'] = "lcurve.out"
        if do_init_model:
            odict['learning_rate']['start_lr'] = config['init_model_start_lr']
            odict['loss']['start_pref_e'] = config['init_model_start_pref_e']
            odict['loss']['start_pref_f'] = config['init_model_start_pref_f']
            odict['loss']['start_pref_v'] = config['init_model_start_pref_v']
            if major_version == "1":
                odict['training']['stop_batch'] = config['init_model_numb_steps']
            elif major_version == "2":
                odict['training']['numb_steps'] = config['init_model_numb_steps']
            else:
                raise RuntimeError('unsupported DeePMD-kit major version', major_version)
        return odict

    @staticmethod
    def decide_init_model(
            config,
            init_model,
            init_data,
            iter_data,
    ):
        do_init_model = False
        # decide if we do init-model
        ## cases we do definitely not
        if init_model is None or \
           iter_data is None or \
           len(iter_data) == 0 : 
            do_init_model = False
        ## cases controlled by the policy
        else:
            if config['init_model_policy'] == 'no':
                do_init_model = False
            elif config['init_model_policy'] == 'yes':
                do_init_model = True
            elif 'old_data_larger_than' in config['init_model_policy']:
                old_data_size_level = int(config['init_model_policy'].split(':')[-1])
                init_data_size = _get_data_size_of_all_systems(init_data)
                iter_data_old_size = _get_data_size_of_all_mult_sys(iter_data[:-1])
                old_data_size = init_data_size + iter_data_old_size
                if old_data_size > old_data_size_level:
                    do_init_model = True                
        return do_init_model


    @staticmethod
    def training_args():
        doc_init_model_prolicy = "The policy of init-model training. It can be\n\n\
    - 'no': No init-model training. Traing from scratch.\n\n\
    - 'yes': Do init-model training.\n\n\
    - 'old_data_larger_than:XXX': Do init-model if the training data size of the previous model is larger than XXX. XXX is an int number."
        doc_init_model_old_ratio = "The frequency ratio of old data over new data"
        doc_init_model_numb_steps = "The number of training steps when init-model"
        doc_init_model_start_lr = "The start learning rate when init-model"
        doc_init_model_start_pref_e = "The start energy prefactor in loss when init-model"
        doc_init_model_start_pref_f = "The start force prefactor in loss when init-model"
        doc_init_model_start_pref_v = "The start virial prefactor in loss when init-model"

        return [
            Argument("init_model_policy", str, optional=True, default='no', doc=doc_init_model_prolicy),
            Argument("init_model_old_ratio", float, optional=True, default=0.9, doc=doc_init_model_old_ratio),
            Argument("init_model_numb_steps", int, optional=True, default=400000, doc=doc_init_model_numb_steps, alias = ['init_model_stop_batch']),
            Argument("init_model_start_lr", float, optional=True, default=1e-4, doc=doc_init_model_start_lr),
            Argument("init_model_start_pref_e", float, optional=True, default=0.1, doc=doc_init_model_start_pref_e),
            Argument("init_model_start_pref_f", float, optional=True, default=100, doc=doc_init_model_start_pref_f),
            Argument("init_model_start_pref_v", float, optional=True, default=0.0, doc=doc_init_model_start_pref_v),
        ]
        

    @staticmethod
    def normalize_config(data = {}):
        ta = RunDPTrain.training_args()

        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=True)

        return data


def _get_data_size_of_system(data_dir):
    ss = dpdata.System(data_dir, fmt='deepmd/npy')
    return ss.get_nframes()

def _get_data_size_of_all_systems(data_dirs):
    count = 0
    for ii in data_dirs:
        count += _get_data_size_of_system(ii)
    return count

def _get_data_size_of_mult_sys(data_dir):
    ms = dpdata.MultiSystems()
    ms.from_deepmd_npy(data_dir)
    return ms.get_nframes()

def _get_data_size_of_all_mult_sys(data_dirs):
    count = 0
    for ii in data_dirs:
        count += _get_data_size_of_mult_sys(ii)
    return count

def _expand_multi_sys_to_sys(multi_sys_dir):
    all_type_raws = sorted(glob.glob(os.path.join(multi_sys_dir, '*', 'type.raw')))
    all_sys_dirs = [ str(Path(ii).parent) for ii in all_type_raws ]
    return all_sys_dirs

def _expand_all_multi_sys_to_sys(list_multi_sys):
    all_sys_dirs = []
    for ii in list_multi_sys:
        all_sys_dirs = all_sys_dirs + _expand_multi_sys_to_sys(ii)
    return all_sys_dirs
