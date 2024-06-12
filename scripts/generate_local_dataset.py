from glob2 import glob
from pathlib import Path
import pandas as pd
import pathlib
import traceback
from ribophene.file_io import extract_list, update_akseg_paths, check_channel_list
package_root = Path(__file__).resolve().parents[1]
import pickle
import tifffile
import shutil
from multiprocessing import Pool
import tqdm
import numpy as np


def generate_json_path(path):

    try:

        path = pathlib.Path(path)
        path = path.with_suffix(".txt")

        parts = list(path.parts)
        parts[-3] = "json"

        path = pathlib.Path("").joinpath(*parts)

    except:
        print(traceback.format_exc())
        pass

    return path


def generate_ribophene_path(path, DATA_DIRECTORY):

    try:
        DATA_DIRECTORY = Path(DATA_DIRECTORY)
        parts = list(path.parts)

        file_extension = parts[-1].split(".")[-1]

        if file_extension == "txt":
            path = pathlib.Path("").joinpath(*DATA_DIRECTORY.parts, "labels", *parts[-2:])
        else:
            path = pathlib.Path("").joinpath(*DATA_DIRECTORY.parts, "images", *parts[-2:])

    except:
        print(traceback.format_exc())
        pass

    return path



def read_metadata(path, AKSEG_DIRECTORY, USER_INITIAL, DATA_DIRECTORY):

    metadata = pd.read_csv(path, sep=",", low_memory=False)

    metadata["file_list"] = metadata["file_list"].apply(lambda data: extract_list(data))
    metadata["channel_list"] = metadata["channel_list"].apply(lambda data: extract_list(data, mode="channel"))

    metadata["database_image_path"] = metadata["image_save_path"].apply(lambda path: update_akseg_paths(path, AKSEG_DIRECTORY, USER_INITIAL))
    metadata["database_json_path"] = metadata["database_image_path"].apply(lambda path: generate_json_path(path))

    metadata["local_image_path"] = metadata["database_image_path"].apply(lambda path: generate_ribophene_path(path, DATA_DIRECTORY))
    metadata["local_label_path"] = metadata["database_json_path"].apply(lambda path: generate_ribophene_path(path, DATA_DIRECTORY))

    metadata = metadata.drop_duplicates(subset=['akseg_hash'], keep="first")

    metadata = metadata.drop_duplicates(["segmentation_file", "folder", "content"])
    metadata = metadata[metadata["segmentation_file"] != "missing image channel"]

    metadata = metadata[metadata['channel'].notna()]
    metadata = metadata.reset_index(drop=True)

    return metadata


def move_data(meta):

    try:
        database_image_path = meta["database_image_path"]
        database_json_path = meta["database_json_path"]
        local_image_path = meta["local_image_path"]
        local_label_path = meta["local_label_path"]

        channel_list = meta["channel_list"]
        file_list = meta["file_list"]

        sorted_list = list(zip(channel_list, file_list))
        sorted_list.sort(key=lambda x: x[0])
        channel_list, file_list = zip(*sorted_list)

        images = []

        for file_name in file_list:

            parts = list(Path(database_image_path).parts)
            parts[-1] = file_name
            source_path = Path("").joinpath(*parts)

            image = tifffile.imread(source_path)
            images.append(image)

        images = np.stack(images, axis=0)

        local_image_path = Path(local_image_path)
        image_dir = local_image_path.parents[0]
        image_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(local_image_path, images)

        target_dir = Path(local_label_path).parents[0]
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(database_json_path, local_label_path)

    except:
        print(traceback.format_exc())
        pass

data_dir = package_root / "data"
data_dir = r"D:/ribophene_dataset"

bacseg_database = r"Y:\Piers_2\BacSeg Database"

data_files = glob(data_dir + "/**/*.csv")

metadata_list = [read_metadata(file, bacseg_database, "AF", data_dir) for file in data_files]

# with open(f"{data_dir}/metadata.pkl", "wb") as f:
#     pickle.dump(metadata_list, f)
#
# with open(f"{data_dir}/metadata.pkl", "rb") as f:
#     metadata_list = pickle.load(f)

metadata = pd.concat(metadata_list)
metadata = metadata.drop_duplicates(subset=['segmentation_file'], keep="first")

# extract each row into a list
metadata = metadata.to_dict(orient="records")

if __name__ == '__main__':
    with Pool() as p:
        d = list(tqdm.tqdm(p.imap(move_data,metadata), total=len(metadata)))
        p.close()
        p.join()






