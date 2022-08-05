import unittest

from tensorflow import keras

from models import models


class TestResnet50(unittest.TestCase):
    def test_create_model(self):
        weights = "imagenet"
        shape = (5, 128, 128, 3)
        dropout_rate = 0.1
        trainable_layers = 4
        nbout = 3
        model = models.action_model(
            weights=weights,
            shape=shape,
            dropout_rate=dropout_rate,
            trainable_layers=trainable_layers,
            nbout=nbout,
        )

        # Validate output model is a keras.Model
        self.assertTrue(
            isinstance(model, keras.Model), msg="Output model is not a keras.Model",
        )

        # Validate first layer is ok
        input_layer = model.layers[0]
        self.assertTrue(
            isinstance(input_layer, keras.layers.TimeDistributed),
            msg="Invalid type for model input layer",
        )
        self.assertEqual(
            input_layer.input_shape[1:], shape, msg="Input layer shape is invalid",
        )

        # Check output shape is ok
        self.assertEqual(input_layer.output.shape.as_list(), [None, 5, 2048])

        # Validate LSTM layer is ok
        lstm_layer = model.layers[1]
        self.assertTrue(
            isinstance(lstm_layer, keras.layers.LSTM),
            msg="Invalid type for model LSTM layer",
        )

        # Check Droupout layer is present in the model
        dropout_layer = model.layers[-2]
        self.assertTrue(
            isinstance(dropout_layer, keras.layers.Dropout),
            msg="Dropout layer not found or incorrectly placed",
        )
        #  Validate dropout rate is ok
        self.assertAlmostEqual(
            dropout_layer.rate,
            dropout_rate,
            places=4,
            msg="Dropout rate not being applied",
        )

        # Check full model output
        output_layer = model.layers[-1]
        self.assertTrue(
            isinstance(output_layer, keras.layers.Dense),
            msg="Output layer not found or incorrectly placed",
        )
        # Check output shape is ok
        self.assertEqual(
            output_layer.output.shape.as_list(),
            [None, nbout],
            msg="Number of output model classes is incorrect",
        )
        # Check output activation function
        self.assertTrue(
            output_layer.activation == keras.activations.softmax,
            msg="Model output activation must be Softmax",
        )


if __name__ == "__main__":
    unittest.main()
