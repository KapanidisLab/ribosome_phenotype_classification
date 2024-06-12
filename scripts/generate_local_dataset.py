import os.path

from glob2 import glob
from pathlib import Path
import pandas as pd
import pathlib
import traceback
from ribophene.file_io import extract_list
import pickle
import tifffile
import shutil
from multiprocessing import Pool
import tqdm
import numpy as np
import random
import string
from ribophene.file_io import import_coco_json

def update_ribophene_paths(row, AKSEG_DIRECTORY, USER_INITIAL, DATA_DIRECTORY):

    try:

        path = row["image_save_path"]
        segmentation_file = row["segmentation_file"]
        folder = row["folder"]
        json_file = segmentation_file.replace("tif", "txt")

        path = pathlib.Path(path.replace("\\","/"))
        AKSEG_DIRECTORY = pathlib.Path(AKSEG_DIRECTORY)

        index = path.parts.index(str(USER_INITIAL))

        database_image_parts = (*AKSEG_DIRECTORY.parts, "Images", *path.parts[index:])
        database_image_path = pathlib.Path('').joinpath(*database_image_parts)

        database_label_parts = list(database_image_path.parts)
        database_label_parts[-3] = "json"
        database_label_parts[-1] = json_file
        database_json_path = pathlib.Path('').joinpath(*database_label_parts)

        DATA_DIRECTORY = Path(DATA_DIRECTORY)
        random_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        image_name = random_name + ".tif"
        label_name = random_name + ".tif"

        local_image_path = pathlib.Path("").joinpath(*DATA_DIRECTORY.parts, "images", image_name)
        local_label_path = pathlib.Path("").joinpath(*DATA_DIRECTORY.parts, "labels", label_name)

        row["database_image_path"] = database_image_path
        row["database_json_path"] = database_json_path
        row["image_path"] = local_image_path
        row["label_path"] = local_label_path
        row["image_name"] = image_name
        row["label_name"] = label_name

    except:
        print(traceback.format_exc())
        pass

    return row

def read_metadata(path, AKSEG_DIRECTORY, USER_INITIAL, DATA_DIRECTORY):

    metadata = pd.read_csv(path, sep=",", low_memory=False)

    metadata["file_list"] = metadata["file_list"].apply(lambda data: extract_list(data))
    metadata["channel_list"] = metadata["channel_list"].apply(lambda data: extract_list(data, mode="channel"))

    metadata = metadata.apply(lambda row: update_ribophene_paths(row, AKSEG_DIRECTORY,
        USER_INITIAL, DATA_DIRECTORY), axis=1)

    metadata = metadata.drop_duplicates(subset=['akseg_hash'], keep="first")

    metadata = metadata.drop_duplicates(["segmentation_file", "folder", "content"])
    metadata = metadata[metadata["segmentation_file"] != "missing image channel"]

    metadata = metadata[metadata['channel'].notna()]
    metadata = metadata.reset_index(drop=True)

    return metadata


def move_data(job):

    try:

        meta, ribophene_meta = job

        database_image_path = meta["database_image_path"]
        database_json_path = meta["database_json_path"]
        local_image_path = meta["image_path"]
        local_label_path = meta["label_path"]

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
        tifffile.imwrite(local_image_path, images,
            metadata=ribophene_meta)

        target_dir = Path(local_label_path).parents[0]
        target_dir.mkdir(parents=True, exist_ok=True)
        mask, _ = import_coco_json(database_json_path, ["single"])
        tifffile.imwrite(local_label_path, mask)

    except:
        print(traceback.format_exc())
        pass

def obfuscate_strain(strain):

    strain_metadata = {"L52042": "S1",
                       "L48480": "S2",
                       "L74177": "S3",
                       "L23827": "R1",
                       "L60725": "R2",
                       "L55048": "R3",
                       "L17667": "R4"}

    if strain in strain_metadata.keys():
        strain = strain_metadata[strain]

    return strain



package_root = Path(__file__).resolve().parents[1]
data_dir = package_root / "data"

data_dir = r"D:/ribophene_dataset"

ribophene_columns = ['dataset','antibiotic','image_name', 'label_name', 'channel_list',
                     'strain','antibiotic concentration','treatment time (mins)',
                     'user_meta1','user_meta3']

bacseg_database = r"Y:\Piers_2\BacSeg Database"
data_files = glob(data_dir + "/**/*.csv")

if __name__ == '__main__':

    for file in data_files:

        if "_ribophene" in file:
            continue

        file_name = os.path.basename(file)

        print(f"Processing {file_name}")

        metadata = read_metadata(file, bacseg_database, "AF", data_dir)
        metadata = metadata.drop_duplicates(subset=['segmentation_file'], keep="first")

        ribophene_metadata = metadata[ribophene_columns].copy()
        ribophene_metadata.rename(columns={"antibiotic": "label",
                                           "user_meta3": "BioRep"}, inplace=True)

        if file_name in ["metadata_MG1655_cam_testD.csv",
                         "metadata_MG1655_cip_testD.csv",
                         "metadata_MG1655_gent_testD.csv"]:

            ribophene_metadata["strain"] = "MG1655"
        else:
            ribophene_metadata["strain"] = ribophene_metadata["user_meta1"]

        ribophene_metadata["strain"] = ribophene_metadata["strain"].apply(obfuscate_strain)
        ribophene_metadata.drop(columns=["user_meta1"], inplace=True)

        export_path = file.split("_")[:-1]
        export_path = "_".join(export_path) + "_ribophene.csv"
        export_path = os.path.normpath(export_path)
        ribophene_metadata.to_csv(export_path, index=False)

        metadata = metadata.to_dict(orient="records")
        ribophene_metadata = ribophene_metadata.to_dict(orient="records")

        upload_jobs = list(zip(metadata, ribophene_metadata))

        with Pool() as p:
            d = list(tqdm.tqdm(p.imap(move_data, upload_jobs), total=len(upload_jobs)))
            p.close()
            p.join()

