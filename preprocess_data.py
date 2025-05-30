import os
from read_bvh import BvhMocap
from utils import read_pickle, save_pickle
from tqdm.contrib import tenumerate


def traverse_path(path, filetype, files):
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            traverse_path(os.path.join(dirpath, dirname), filetype, files)
        for filename in filenames:
            if not filename.endswith(filetype):
                continue
            files.append(os.path.join(dirpath, filename))


def preprocess_data(dataset_folder, data_cache_path, save_every=10):
    bvh_files = []
    traverse_path(dataset_folder, ".bvh", bvh_files)
    bvh_files = sorted(list(set(bvh_files)))
    data = {}
    if os.path.exists(data_cache_path):
        data = read_pickle(data_cache_path)
        data = {key: data[key] for key in data.keys()}
    bvh_files = [file for file in bvh_files if file not in data]
    for index, bvh_file in tenumerate(bvh_files):
        mocap_object = BvhMocap(bvh_file)
        data[bvh_file] = mocap_object.export_array()
        if index % save_every == 0:
            save_pickle(data_cache_path, data)
    save_pickle(data_cache_path, data)


if __name__ == "__main__":
    from utils import read_json
    config_path = "config.json"
    config = read_json(config_path)
    preprocess_data(dataset_folder=os.path.join("datasets", config["dataset"]),
                    data_cache_path=config["data_cache_path"],
                    save_every=10)
