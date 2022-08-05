from sklearn import preprocessing
from tensorflow import keras
import pandas as pd

# Supported optimizer algorithms
OPTIMIZERS = {
    "adam": keras.optimizers.Adam,
    "sgd": keras.optimizers.SGD,
}


# Supported callbacks
CALLBACKS = {
    "model_checkpoint": keras.callbacks.ModelCheckpoint,
    "tensor_board": keras.callbacks.TensorBoard,
    "reduce_on_plateau": keras.callbacks.ReduceLROnPlateau,
}


def parse_optimizer(config):
    """
    Get experiment settings for optimizer algorithm.

    Parameters
    ----------
    config : str
        Experiment settings.
    """
    opt_name, opt_params = list(config["compile"]["optimizer"].items())[0]
    optimizer = OPTIMIZERS[opt_name](**opt_params)

    del config["compile"]["optimizer"]

    return optimizer


def parse_callbacks(config):
    """
    Add Keras callbacks based on experiment settings.

    Parameters
    ----------
    config : str
        Experiment settings.
    """
    callbacks = []
    if "callbacks" in config["fit"]:
        for callbk_name, callbk_params in config["fit"]["callbacks"].items():
            callbacks.append(CALLBACKS[callbk_name](**callbk_params))

        del config["fit"]["callbacks"]

    return callbacks


def encode_csv(config):

    # open csv
    train_df = pd.read_csv(config["data"]["train_df"], index_col="Unnamed: 0")

    val_df = pd.read_csv(config["data"]["val_df"], index_col="Unnamed: 0")

    # shuffle the samples
    train_df = train_df.sample(frac=1)

    print(f"Total videos for training: {len(train_df)}")
    print(f"Total videos for validation: {len(val_df)}")

    le_movement = preprocessing.LabelEncoder()
    le_scale = preprocessing.LabelEncoder()

    train_df["movement"] = le_movement.fit_transform(train_df["movement"])
    train_df["scale"] = le_scale.fit_transform(train_df["scale"])

    val_df["movement"] = le_movement.transform(val_df["movement"])
    val_df["scale"] = le_scale.transform(val_df["scale"])

    return train_df, val_df


def train_model(config, model, training_generator, validation_generator):

    optimizer = parse_optimizer(config)

    model.compile(
        optimizer=optimizer, **config["compile"],
    )

    callbacks = parse_callbacks(config)

    model.fit(
        training_generator,
        validation_data=validation_generator,
        verbose=1,
        callbacks=callbacks,
        **config["fit"],
    )
