import os
import cv2
from random import sample
from tensorflow import keras
from yaml import load, SafeLoader
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


IMG_SIZE = 224


def validate_config(config):
    """
    Takes as input the experiment configuration as a dict and checks for
    minimum acceptance requirements.

    Parameters
    ----------
    config : dict
        Experiment settings as a Python dict.
    """
    if "seed" not in config:
        raise ValueError("Missing experiment seed")

    if "data" not in config:
        raise ValueError("Missing experiment data")

    if "train_df" not in config["data"]:
        raise ValueError("Missing experiment training data")


def load_config(config_file_path):
    """
    Loads experiment settings from a YAML file into a Python dict.
    See: https://pyyaml.org/.

    Parameters
    ----------
    config_file_path : str
        Full path to experiment configuration file.
        E.g: `/home/app/src/experiments/exp_001/config.yml`

    Returns
    -------
    config : dict
        Experiment settings as a Python dict.
    """
    with open(config_file_path, "r") as f:
        stream = load(f, Loader=SafeLoader)

    config = stream

    # Don't remove this as will help you doing some basic checks on config
    # content
    validate_config(config)

    return config


def get_class_names(config):
    """

    Parameters
    ----------
    config : dict
        Experiment settings as Python dict.

    Returns
    -------
    classes : list
        List of classes as string.
        E.g. ['AM General Hummer SUV 2000', 'Buick Verano Sedan 2012',
                'FIAT 500 Abarth 2012', 'Jeep Patriot SUV 2012',
                'Acura Integra Type R 2001', ...]
    """
    train_df = pd.read_csv(config["data"]["train_df"], index_col="Unnamed: 0")
    movement_names = sorted(list(train_df.movement.unique()))
    scale_names = sorted(list(train_df.scale.unique()))
    return movement_names, scale_names


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


def load_video(
    path, inversed_frame, random_sample=False, max_frames=8, resize=(IMG_SIZE, IMG_SIZE)
):
    numbers_frames = []
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if random_sample:
        for i in range(length):
            numbers_frames.append(i)
        sample_frames = sorted(sample(numbers_frames, max_frames))

    else:
        sample_frames = np.linspace(0, length - 1, max_frames, dtype=int)

    frames = []
    count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count in sample_frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, resize)

                preprocess_input = keras.applications.resnet50.preprocess_input
                frame = preprocess_input(frame)

                frames.append(frame)
            count += 1

    finally:
        cap.release()

    # reverse only pull videos
    if inversed_frame == "yes":
        frames = frames[::-1]

    return np.array(frames)


def create_list_v2(path_json):
    with open(path_json, "rb") as f:
        json_file = f.read()
        json_file = json.loads(json_file)

    movement_train = []

    ubications_train = []

    root_ubication = "../data/data_cleaned/v2/"

    for movie in json_file:
        for shot in json_file[movie]:
            for label in json_file[movie][shot]:
                value = json_file[movie][shot][label]["label"]
                ubication = os.path.join(root_ubication, movie)
                ubication = ubication + "_shot_" + shot + ".mp4"

                movement_train.append(value)
                ubications_train.append(ubication)
    return movement_train, ubications_train


def get_performance(predictions, y_test, labels):
    accuracy = metrics.accuracy_score(y_test, predictions)
    precision = metrics.precision_score(y_test, predictions, average="macro")
    recall = metrics.recall_score(y_test, predictions, average="macro")
    f1_score = metrics.f1_score(y_test, predictions, average="macro")

    report = metrics.classification_report(y_test, predictions)

    cm = metrics.confusion_matrix(y_test, predictions, normalize="true")
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    print("Model Performance metrics:")
    print("-" * 30)
    print("Accuracy:", round(accuracy, 4))
    print("Macro Average Precision:", round(precision, 4))
    print("Macro Average Recall:", round(recall, 4))
    print("Macro Average F1 Score:", round(f1_score, 4))
    print("\nModel Classification report:")
    print("-" * 30)
    print(report)
    print("\nPrediction Confusion Matrix:")
    print("-" * 30)
    disp.plot()
    disp.ax_.get_images()[0].set_clim(0, 1)
    plt.show()

    return accuracy, precision, recall, f1_score
