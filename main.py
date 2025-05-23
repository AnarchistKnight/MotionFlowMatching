import os
from tqdm import tqdm
from read_bvh import BvhMocap
import numpy as np

BVH_FOLDER = "datasets"


def main():
    motion_data = {}
    dataset_names = os.listdir(BVH_FOLDER)
    for dataset_name in dataset_names:
        dataset_folder = os.path.join(BVH_FOLDER, dataset_name)
        for dirpath, _, filenames in os.walk(dataset_folder):
            for filename in filenames:
                if not filename.endswith(".bvh"):
                    continue
                bvh_file_path = os.path.join(dirpath, filename)
                mocap_object = BvhMocap(bvh_file_path)
                print("="*20, dataset_name, "="*20)
                print(mocap_object.joint_names)
                # mocap_object.visualize(10, 600, 2)
                break
            break
    # np.savez("data.npz", **motion_data)


if __name__ == "__main__":
    main()