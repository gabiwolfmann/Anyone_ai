from tensorflow import keras
from tensorflow.keras.utils import Sequence
import math
from utils import utils
import numpy as np
from keras.layers import TimeDistributed, Dense, Dropout, LSTM


class DataGenerator(Sequence):
    def __init__(self, df, batch_size, output, random_sample=False):
        self.df = df
        self.batch_size = batch_size
        self.num_samples = len(df)
        self.video_paths = df["ubication"].values.tolist()
        self.movements_labels = df["movement"].values.tolist()
        self.scale_labels = df["scale"].values.tolist()
        if not "inversed" in df:
            df["inversed"] = "no"
        self.inversed = df["inversed"].values.tolist()
        self.random_sample = random_sample
        self.output = output

    def __len__(self):
        return math.ceil(self.df.shape[0] / self.batch_size)

    def __getitem__(self, batch_index):
        batch_movement_label = []
        batch_scale_label = []
        batch_frame_features = []

        for idx, path in enumerate(self.video_paths):

            if (batch_index * self.batch_size) > idx:
                continue
            if (batch_index + 1) * self.batch_size <= idx:
                break

            # Gather all its frames and add a batch dimension.
            inversed_frame = self.inversed[idx]
            frames = utils.load_video(path, inversed_frame, self.random_sample)

            batch_movement_label.append(self.movements_labels[idx])
            batch_scale_label.append(self.scale_labels[idx])

            batch_frame_features.append(frames)

        if self.output == "movement":
            labels = np.array(batch_movement_label)

        elif self.output == "scale":
            labels = np.array(batch_scale_label)

        elif self.output == "both":
            labels = [np.array(batch_movement_label), np.array(batch_scale_label)]

        else:
            raise ValueError(
                "Insert the correct variable name: 'movement', 'scale' or 'both' "
            )

        return np.array(batch_frame_features), labels


def build_feature_extractor(shape, trainable_layers=None):
    feature_extractor = keras.applications.ResNet50(
        weights="imagenet", include_top=False, pooling="avg", input_shape=shape,
    )
    if trainable_layers is not None:
        for layer in feature_extractor.layers[:-trainable_layers]:
            layer.trainable = False

    return feature_extractor


def action_model(
    shape=(8, 224, 224, 3),
    nbout=4,
    weights="imagenet",
    trainable_layers=None,
    dropout_rate=0,
):

    if weights == "imagenet":
        # Create our convnet
        convnet = build_feature_extractor(shape[1:], trainable_layers)

        # then create our final model
        model = keras.Sequential()
        # add the convnet
        model.add(TimeDistributed(convnet, input_shape=shape))
        # here, you can also use GRU or LSTM
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(16))
        # and finally, we make a decision network
        model.add(Dropout(dropout_rate))
        model.add(Dense(64, activation="relu", kernel_regularizer="l2"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(nbout, activation="softmax", kernel_regularizer="l2"))

    return model


def action_model_multi_output(
    shape=(8, 224, 224, 3),
    n_movement=4,
    n_scale=5,
    weights="imagenet",
    trainable_layers=None,
    dropout_rate=0,
):

    if weights == "imagenet":
        convnet = build_feature_extractor(shape[1:], trainable_layers)

        inputs = keras.Input(shape=shape, dtype="float32")
        # add the convnet with (5, 112, 112, 3) shape
        x = TimeDistributed(convnet)(inputs)
        # here, you can also use GRU or LSTM
        x_mov = LSTM(256, return_sequences=True)(x)
        x_mov = LSTM(128)(x_mov)
        x_mov = Dropout(dropout_rate)(x_mov)
        x_mov = Dense(64, activation="relu", kernel_regularizer="l2")(x_mov)
        x_mov = Dropout(dropout_rate)(x_mov)
        # and finally, we make a decision network
        outputs_movement = Dense(
            n_movement, activation="softmax", kernel_regularizer="l2", name="movement"
        )(x_mov)

        x_scale = LSTM(256, return_sequences=True)(x)
        x_scale = LSTM(128)(x_scale)
        x_scale = Dropout(dropout_rate)(x_scale)
        x_scale = Dense(64, activation="relu", kernel_regularizer="l2")(x_scale)
        x_scale = Dropout(dropout_rate)(x_scale)
        outputs_scale = Dense(
            n_scale, activation="softmax", kernel_regularizer="l2", name="scale"
        )(x_scale)

        model = keras.Model(inputs, [outputs_movement, outputs_scale])

    return model
