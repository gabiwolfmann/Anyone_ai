"""
This script will be used for training our CNN + RNN for movement target. The only input argument it
should receive is the path to our configuration file in which we define all
the experiment settings like dataset, model output folder, epochs,
learning rate, etc.
"""

from utils import utils, train_utils
from models import models
import argparse


def parse_args():
    """
    Use argparse to get the input parameters for training the model.
    """
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "config_file", type=str, help="Full path to experiment configuration file.",
    )

    args = parser.parse_args()

    return args


def main(config_file):
    """
    Code for the training logic.

    Parameters
    ----------
    config_file : str
        Full path to experiment configuration file.
    """
    # Load configuration file.
    config = utils.load_config(config_file)
    movement_names, scale_names = utils.get_class_names(config)

    # Check if number of classes is correct
    if len(movement_names) != config["model"]["classes_movement"]:
        raise ValueError(
            "The number classes between your dataset and your model" "doesn't match."
        )
    # Check if number of classes is correct
    if len(scale_names) != config["model"]["classes_scale"]:
        raise ValueError(
            "The number classes between your dataset and your model" "doesn't match."
        )

    train_df, val_df = train_utils.encode_csv(config)

    # instantiate the data generator
    training_generator = models.DataGenerator(
        train_df, config["model"]["batch_size"], output="movement"
    )
    validation_generator = models.DataGenerator(
        val_df, config["model"]["batch_size"], output="movement"
    )

    INSHAPE = (
        (config["model"]["frames"],)
        + (config["model"]["image_size"],)
        + (config["model"]["image_size"],)
        + (config["model"]["channels"],)
    )  # (8, 224, 224, 3)
    model = models.action_model(
        INSHAPE,
        len(movement_names),
        trainable_layers=config["model"]["trainable_layers"],
        dropout_rate=config["model"]["dropout_rate"],
    )
    print(model.summary())

    train_utils.train_model(config, model, training_generator, validation_generator)


if __name__ == "__main__":
    args = parse_args()
    main(args.config_file)
