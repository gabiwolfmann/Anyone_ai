"""
This script will be used to separate and copy images coming from
`car_ims.tgz` (extract the .tgz content first) between `train` and `test`
folders according to the column `subset` from `car_dataset_labels.csv`.
It will also create all the needed subfolders inside `train`/`test` in order
to copy each image to the folder corresponding to its class.

The resulting directory structure should look like this:
    data/
    ├── car_dataset_labels.csv
    ├── car_ims
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   ├── ...
    ├── car_ims_v1
    │   ├── test
    │   │   ├── AM General Hummer SUV 2000
    │   │   │   ├── 000046.jpg
    │   │   │   ├── 000047.jpg
    │   │   │   ├── ...
    │   │   ├── Acura Integra Type R 2001
    │   │   │   ├── 000450.jpg
    │   │   │   ├── 000451.jpg
    │   │   │   ├── ...
    │   ├── train
    │   │   ├── AM General Hummer SUV 2000
    │   │   │   ├── 000001.jpg
    │   │   │   ├── 000002.jpg
    │   │   │   ├── ...
    │   │   ├── Acura Integra Type R 2001
    │   │   │   ├── 000405.jpg
    │   │   │   ├── 000406.jpg
    │   │   │   ├── ...
"""
import argparse
import os
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. E.g. "
            "`/home/app/src/data/car_ims/`."
        ),
    )
    parser.add_argument(
        "labels",
        type=str,
        help=(
            "Full path to the CSV file with data labels. E.g. "
            "`/home/app/src/data/car_dataset_labels.csv`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "train/test splits. E.g. `/home/app/src/data/car_ims_v1/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, labels, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to raw images folder.

    labels : str
        Full path to CSV file with data annotations.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        train/test splits.
    """
    # For this function, you must:
    #   1. Load labels CSV file
    #   2. Iterate over each row in the CSV, create the corresponding
    #      train/test and class folders
    #   3. Copy the image to the new folder structure. We recommend you to
    #      use `os.link()` to avoid wasting disk space with duplicated files
    # TODO

    df_labels = pd.read_csv(labels)

    os.makedirs(output_data_folder, exist_ok=True)

    path_train = os.path.join(output_data_folder, "train")
    os.makedirs(path_train, exist_ok=True)

    path_test = os.path.join(output_data_folder, "test")
    os.makedirs(path_test, exist_ok=True)

    for _, row in tqdm(df_labels.iterrows()):
        if row.subset == "train":

            path_label_train = os.path.join(path_train, row["class"])
            path_input_train = os.path.join(
                data_folder, row["img_name"]
            )
            path_output_train = os.path.join(
                path_label_train, row["img_name"]
            )

            os.makedirs(path_label_train, exist_ok=True)

            if not os.path.exists(path_output_train):
                os.link(path_input_train, path_output_train)

        else:

            path_label_test = os.path.join(path_test, row["class"])
            path_input_test = os.path.join(
                data_folder, row["img_name"]
            )
            path_output_test = os.path.join(
                path_label_test, row["img_name"]
            )

            os.makedirs(path_label_test, exist_ok=True)

            if not os.path.exists(path_output_test):
                os.link(path_input_test, path_output_test)


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.labels, args.output_data_folder)
