import json
import pickle
import bvh


def read_bvh(bvh_file):
    with open(bvh_file, "r") as f:
        return bvh.Bvh(f.read())


def read_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def read_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def save_pickle(pickle_path, data):
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f)