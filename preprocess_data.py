import os
from tqdm import tqdm
from read_bvh import BvhMocap
import pickle

BVH_FOLDER = "datasets"
DATASET_NAME = "100STYLE"
SAVE_EVERY = 10


def traverse_path(path, filetype, files):
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            traverse_path(os.path.join(dirpath, dirname), filetype, files)
        for filename in filenames:
            if not filename.endswith(filetype):
                continue
            files.append(os.path.join(dirpath, filename))


def main():
    dataset_folder = os.path.join(BVH_FOLDER, DATASET_NAME)
    bvh_files = []
    traverse_path(dataset_folder, ".bvh", bvh_files)
    bvh_files = sorted(list(set(bvh_files)))
    data_path = "data.pkl"
    data = {}
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            data = {key: data[key].reshape(-1, 23, 9) for key in data.keys()}
            print([data[key].shape for key in data.keys()])
            print(len(data), "files already processed and cached")
    index = 0
    for bvh_file_path in tqdm(bvh_files):
        if bvh_file_path in data.keys():
            continue
        mocap_object = BvhMocap(bvh_file_path)
        motion_array = mocap_object.export_array()
        data[bvh_file_path] = motion_array
        index += 1
        if index % SAVE_EVERY == 0:
            with open(data_path, "wb") as f:
                pickle.dump(data, f)
    with open(data_path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
