import pickle
from pathlib import Path

def dump_object_to_file(
        obj,
        fname,
):
    """
    pickle dump object to a file

    """
    with open(fname, 'wb') as fp:
        pickle.dump(obj, fp)
    return Path(fname)

def load_object_from_file(
        fname,
):
    """
    pickle load object from a file

    """
    with open(fname, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

        
