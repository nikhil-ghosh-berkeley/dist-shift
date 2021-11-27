import pickle
import fcntl

def load_pickle(path):
    with open(path, "rb") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        ret = pickle.load(f)
        fcntl.flock(f, fcntl.LOCK_UN)
    return ret

def dump_pickle(obj, path):
    with open(path, "wb") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        ret = pickle.dump(obj, f)
        fcntl.flock(f, fcntl.LOCK_UN)
    return ret

