import os
import cv2
import random
import pandas as pd
from moviepy.editor import VideoFileClip
from moviepy.editor import CompositeVideoClip


# static_to_push.csv is a csv with only the films that are about to be changed,
# only care about 'shot_id' and 'film_id'.
changed_df = pd.read_csv("data/CSV/static_to_push.csv", dtype={"shot_id": "str"})

for index, row in changed_df.iterrows():

    file_name = "shot_" + row["shot_id"] + ".mp4"
    video_path = os.path.join("/home/data-proyect/trailer", row["film_id"], file_name)
    if not os.path.exists(video_path):
        continue
    new_name = os.path.join(
        "/home/data-proyect/trailer_zoom", (row["film_id"] + "_" + file_name)
    )
    if os.path.exists(new_name):
        continue

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    screensize = (width, height)

    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    duration = frames / frame_rate

    random_zoom = random.uniform(0.1, 0.3)

    clip_img = (
        VideoFileClip(video_path)
        .resize(screensize)
        .resize(lambda t: 1 + random_zoom * t)
        .set_position(("center", "center"))
        .set_duration(duration)
    )

    clip = CompositeVideoClip([clip_img], size=screensize)
    clip.write_videofile(new_name)
