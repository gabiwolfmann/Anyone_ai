"""
This script will be used to create a video links with zoom
to their correspondent folder data/data_cleaned/zoom/
in order to have access inside the container and make
test with them.
"""
import pandas as pd
import os

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "csv_path",
        type=str,
        help=(
            "Full path to the directory having all the annotations. E.g. "
            "`'data/final_static_to_zoom.csv'`."
        ),
    )

    args = parser.parse_args()

    return args


def main(csv_path):
    zoom_changes = pd.read_csv(csv_path)
    root_output_ubication = "data/data_cleaned/zoom/"
    for idx, row in zoom_changes.iterrows():
        video_input_path = row.path
        shot_name = os.path.basename(video_input_path)
        output_ubication = os.path.join(root_output_ubication, shot_name)
        if not os.path.exists(output_ubication):
            os.link(video_input_path, output_ubication)


if __name__ == "__main__":
    args = parse_args()
    main(args.csv_path)
