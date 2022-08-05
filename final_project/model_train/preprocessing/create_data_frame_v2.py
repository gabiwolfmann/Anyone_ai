"""
This script will be used to create a csv with the information of 
v2 videos according to  `v2_full_trailer.json`.
In order to get more labels in target push and pull.

"""
from utils import utils
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "json_v2",
        type=str,
        help=(
            "Full path to the directory having all the annotations. E.g. "
            "`'/home/app/src/data/v2_full_trailer.json'`."
        ),
    )

    args = parser.parse_args()

    return args


def main(json_v2):

    movement_train, ubications_train = utils.create_list_v2(json_v2)
    dict_train_v2 = {"movement": movement_train, "ubication": ubications_train}
    train_df_v2 = pd.DataFrame(dict_train_v2)
    train_df_v2.to_csv("/home/app/src/data/train_df_v2.csv")


if __name__ == "__main__":
    args = parse_args()
    main(args.json_v2)
