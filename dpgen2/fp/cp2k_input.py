from pathlib import (
    Path,
)
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import dpdata
import numpy as np
from dargs import (
    Argument,
    ArgumentEncoder,
    Variant,
    dargs,
)


default_config = {
    "GLOBAL": {"PROJECT": "DPGEN"},
    "FORCE_EVAL": {
        "METHOD": "QS",
        "STRESS_TENSOR": "ANALYTICAL",
        "DFT": {
            "BASIS_SET_FILE_NAME": "BASIS_MOLOPT",
            "POTENTIAL_FILE_NAME": "GTH_POTENTIALS",
            "CHARGE": 0,
            "UKS": "F",
            "MULTIPLICITY": 1,
            "MGRID": {"CUTOFF": 400, "REL_CUTOFF": 50, "NGRIDS": 4},
            "QS": {"EPS_DEFAULT": "1.0E-12"},
            "SCF": {"SCF_GUESS": "ATOMIC", "EPS_SCF": "1.0E-6", "MAX_SCF": 50},
            "XC": {"XC_FUNCTIONAL": {"_": "PBE"}},
        },
        "SUBSYS": {
            "@include": "coord_n_cell.inc",
            "KIND": {
                "_": ["H", "C", "N"],
                "POTENTIAL": ["GTH-PBE-q1", "GTH-PBE-q4", "GTH-PBE-q5"],
                "BASIS_SET": ["DZVP-MOLOPT-GTH", "DZVP-MOLOPT-GTH", "DZVP-MOLOPT-GTH"],
            },
        },
        "PRINT": {"FORCES": {"_": "ON"}, "STRESS_TENSOR": {"_": "ON"}},
    },
}


def update_dict(
    old_d: Dict[str, str],
    update_d: Dict[str, str]
):
    """A method to recursively update a dictionary.

    Parameters
    ----------
    old_d : Dict[str, str]
        The old dictionary to be updated.
    update_d : Dict[str, str]
        The dictionary containing the update values.
    """
    import collections.abc

    for k, v in update_d.items():
        if (
            k in old_d
            and isinstance(old_d[k], dict)
            and isinstance(update_d[k], collections.abc.Mapping)
        ):
            update_dict(old_d[k], update_d[k])
        else:
            old_d[k] = update_d[k]


def iterdict(
    d: Dict[str, str],
    out_list: list,
    flag: Optional[str] = None
):
    """Recursive expansion of dictionary into cp2k input.

    Parameters
    ----------
    d : Dict[str, str]
        Current dictionary under expansion.
    out_list : List[str]
        List to store the expanded cp2k input.
    flag : Optional[str]
        Used to record dictionary state. If flag is None, it means we are in the top level dict.
    """
    for k, v in d.items():
        k = str(k)  # cast key into string
        # if value is dictionary
        if isinstance(v, dict):
            # flag == None, it is now in top level section of cp2k
            if flag is None:
                out_list.append("&" + k)
                out_list.append("&END " + k)
                iterdict(v, out_list, k)
            # flag is not None, now it has name of section
            else:
                index = out_list.index("&END " + flag)
                out_list.insert(index, "&" + k)
                out_list.insert(index + 1, "&END " + k)
                iterdict(v, out_list, k)
        elif isinstance(v, list):
            # print("we have encountered the repeat section!")
            index = out_list.index("&" + flag)
            # delete the current constructed repeat section
            del out_list[index: index + 2]
            # do a loop over key and corresponding list
            k_tmp_list = []
            v_list_tmp_list = []
            for k_tmp, v_tmp in d.items():
                k_tmp_list.append(str(k_tmp))
                v_list_tmp_list.append(v_tmp)
            for repeat_keyword in zip(*v_list_tmp_list):
                out_list.insert(index, "&" + flag)
                out_list.insert(index + 1, "&END " + flag)
                for idx, k_tmp in enumerate(k_tmp_list):
                    if k_tmp == "_":
                        out_list[index] = "&" + flag + \
                            " " + repeat_keyword[idx]
                    else:
                        out_list.insert(index + 1, k_tmp +
                                        " " + repeat_keyword[idx])

            break

        else:
            v = str(v)
            if flag is None:
                out_list.append(k + " " + v)
                print(k, ":", v)
            else:
                if k == "_":
                    index = out_list.index("&" + flag)
                    out_list[index] = "&" + flag + " " + v

                else:
                    index = out_list.index("&END " + flag)
                    out_list.insert(index, k + " " + v)


class Cp2kInputs:
    def __init__(
        self,
        input_template: Dict[str, str],
    ):
        """
        Parameters
        ----------
        input_template : Dict[str, str]
            The dict of CP2K input. Following pattern of cp2k-input-tools.
        """
        self.input_template = input_template

    def make_cp2k_input(self):
        user_config = self.input_template
        update_dict(default_config, user_config)
        # get update from user
        user_config = self.input_template
        # output list
        input_str = []
        iterdict(default_config, input_str)
        string = "\n".join(input_str)
        return string

    @staticmethod
    def make_cp2k_coord_cell(
        sys_data: dpdata.System
    ):
        # get structral information
        atom_names = sys_data["atom_names"]
        atom_types = sys_data["atom_types"]

        # write coordinate to xyz file used by cp2k input
        coord_list = sys_data["coords"][0]
        u = np.array(atom_names)
        atom_list = u[atom_types]
        x = "&COORD\n"
        for kind, coord in zip(atom_list, coord_list):
            x += str(kind) + " " + str(coord[:])[1:-1] + "\n"
        x += "&END COORD\n"

        # covert cell to cell string
        cell = sys_data["cells"][0]
        cell = np.reshape(cell, [3, 3])
        cell_a = np.array2string(cell[0, :])
        cell_a = cell_a[1:-1]
        cell_b = np.array2string(cell[1, :])
        cell_b = cell_b[1:-1]
        cell_c = np.array2string(cell[2, :])
        cell_c = cell_c[1:-1]
        x += "\n"

        x += "&CELL\n"
        x += " A " + cell_a + "\n"
        x += " B " + cell_b + "\n"
        x += " C " + cell_c + "\n"
        x += "&END CELL\n"
        return x

    @staticmethod
    def args():
        doc_input = "The dict of CP2K input. Following pattern of cp2k-input-tools."
        return [
            Argument("input_template", dict, optional=False, doc=doc_input),
        ]

    @staticmethod
    def normalize_config(data={}, strict=True):
        ta = Cp2kInputs.args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=strict)
        return data
