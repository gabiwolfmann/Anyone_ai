import unittest

from utils.utils import load_config


class TestUtils(unittest.TestCase):
    def test_load_config(self):
        config = {
            "seed": 123,
            "data": {
                "train_df": "/bla/data_train",
                "val_df": "/bla/data_val",
                "test_df": "/bla/data_test",
            },
            "model": {
                "weights": "imagenet",
                "classes_movement": 10,
                "classes_scale": 2,
                "dropout_rate": 0.8,
                "image_size": 128,
                "channels": 1,
                "batch_size": 32,
                "frames": 5,
                "trainable_layers": 4,
            },
            "compile": {
                "optimizer": {"adam": {"learning_rate": 0.1}},
                "loss": "categorical_crossentropy",
                "metrics": ["accuracy"],
            },
            "fit": {
                "epochs": 100,
                "callbacks": {
                    "model_checkpoint": {
                        "filepath": "/bla/model",
                        "save_best_only": True,
                    },
                    "tensor_board": {"log_dir": "/bla/log"},
                    "reduce_on_plateau": {"patience": 5},
                },
            },
        }

        loaded_config = load_config("tests/test_data/config_test.yml")

        self.assertDictEqual(loaded_config, config)
