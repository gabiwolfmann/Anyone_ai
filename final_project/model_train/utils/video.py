import json
from utils import utils
import os
import cv2
from tqdm import tqdm


def get_data(path_data):
    f = open(path_data)
    return json.load(f)


def get_video_data(video, path, data, with_frames=True):
    video_root_name = os.path.splitext(os.path.basename(video))[0]
    id_video, shot = video_root_name.split("_shot_")

    categories = ["train", "test", "val"]

    for category in categories:
        if id_video in data[category]:
            if "scale" in data[category][id_video][shot]:
                scale = data[category][id_video][shot]["scale"]["label"]
                scale_value = data[category][id_video][shot]["scale"]["value"]
            else:
                scale = None
                scale_value = None
            movement = data[category][id_video][shot]["movement"]["label"]
            movement_value = data[category][id_video][shot]["movement"]["value"]
            path_file = os.path.join(path, category, os.path.basename(video))
            frames = 0
            fps = 0
            duration = 0
            if with_frames:
                # number frames
                cap = cv2.VideoCapture(path_file)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = frames / fps
            return (
                id_video,
                shot,
                scale,
                scale_value,
                movement,
                movement_value,
                path_file,
                frames,
                fps,
                duration,
            )


def get_videos_with_labels(
    data, max_videos=10, split="train", path="../data/data_cleaned/", with_frames=True
):
    videos_data = []
    for _, filename in tqdm(utils.walkdir(os.path.join(path, split))):
        if len(videos_data) == max_videos:
            break
        videos_data.append(get_video_data(filename, path, data, with_frames))
    return videos_data
