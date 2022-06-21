from tensorflow import keras


def create_data_aug_layer(data_aug_layer):
    """
    Use this function to parse the data augmentation methods for the
    experiment and create the corresponding layers.

    It will be mandatory to support at least the following three data
    augmentation methods (you can add more if you want):
        - `random_flip`: keras.layers.RandomFlip()
        - `random_rotation`: keras.layers.RandomRotation()
        - `random_zoom`: keras.layers.RandomZoom()

    See https://tensorflow.org/tutorials/images/data_augmentation.

    Parameters
    ----------
    data_aug_layer : dict
        Data augmentation settings coming from the experiment YAML config
        file.

    Returns
    -------
    data_augmentation : keras.Sequential
        Sequential model having the data augmentation layers inside.
    """
    # Parse config and create layers
    # You can use as a guide on how to pass config parameters to keras
    # looking at the code in `scripts/train.py`
    # TODO
    # Append the data augmentation layers on this list

    # Supported DATA_AUG
    DATA_AUG = {
        "random_flip": keras.layers.RandomFlip,
        "random_rotation": keras.layers.RandomRotation,
        "random_zoom": keras.layers.RandomZoom,
        "random_contrast": keras.layers.RandomContrast,
    }
    data_aug_layers = []
    if data_aug_layer:  # check if isnt empty dict
        for data_aug_name, data_aug_params in data_aug_layer.items():
            data_aug_layers.append(
                DATA_AUG[data_aug_name](**data_aug_params)
            )

    data_augmentation = keras.Sequential(data_aug_layers)

    # Return a keras.Sequential model having the the new layers created
    # Assign to `data_augmentation` variable
    return data_augmentation
