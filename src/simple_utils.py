import pickle

def load_pickle(path):
    return pickle.load(open(path, "rb"))

def dump_pickle(obj, path):
    return pickle.dump(obj, open(path, "wb"))
