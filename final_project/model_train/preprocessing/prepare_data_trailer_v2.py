"""
This script will be used to create a video copy coming from
(data v2) in a v2 folder according to  `v2_full_trailer.json`.
In order to get more labels in target push and pull.
The resulting directory structure should look like this:
    data/
    ├── v1_full_trailer.json
    ├── v1_split_trailer.json
    ├── data_cleaned
    │   ├── test
    │   ├── train
    │   ├── val
    │   ├── v2
    │   │   ├── tt0042393_shot_0033.mp4
    │   │   ├── tt0042393_shot_0071.mp4
    │   │   ├── ...
"""
from tqdm import tqdm
import os
import json
import argparse


def walkdir(folder):
    """
    Walk through all the files in a directory and its subfolders.

    Parameters
    ----------
    folder : str
        Path to the folder you want to walk.

    Returns
    -------
        For each file found, yields a tuple having the path to the file
        and the file name.
    """
    for dirpath, _, files in os.walk(folder):
        for filename in files:
            yield (dirpath, filename)


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "path_folder_trailer",
        type=str,
        help=(
            "Full path to the directory having all the keyframes. E.g. "
            "`/home/data-proyect/trailer_v2`."
        ),
    )
    parser.add_argument(
        "path_json_split",
        type=str,
        help=(
            "Full path to the directory having all the annotations. E.g. "
            "`data/v2_full_trailer.json`."
        ),
    )
    parser.add_argument(
        "path_output_data",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting keyframes"
            " E.g. `data/data_cleaned/v2/`."
        ),
    )

    args = parser.parse_args()

    return args


def data_cleaning(path_folder_trailer, path_json_split, path_output_data):
    """
    Parameters
    ----------
    path_folder_trailer : str
        Full path to raw videos folder.

    path_json_split : str
        Full path to json file with data annotations.

    path_output_data : str
        Full path to the directory in which we will store the resulting
        train/val/test splits.
    """
    os.makedirs(path_output_data, exist_ok=True)

    with open(path_json_split, "rb") as f:
        json_file = f.read()
        json_file = json.loads(json_file)

    for dirpath, filename in tqdm(walkdir(path_folder_trailer)):
        id_movie = os.path.basename(dirpath)
        video_input_path = os.path.join(dirpath, filename)
        id_shoot = os.path.splitext(filename)[0].split("_")[1]

        # path=os.path.join(path_output_data, category)
        if id_movie in json_file:
            if id_shoot in json_file[id_movie]:
                name_filename = id_movie + "_" + filename
                video_output_path = os.path.join(path_output_data, name_filename)
                if not os.path.exists(video_output_path):
                    os.link(video_input_path, video_output_path)


if __name__ == "__main__":
    args = parse_args()
    data_cleaning(args.path_folder_trailer, args.path_json_split, args.path_output_data)
