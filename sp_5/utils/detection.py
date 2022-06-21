# Load here your Detection model
# The chosen detector model is "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# because this particular model has a good balance between accuracy and speed.
# You can check the following Colab notebook with examples on how to run
# Detectron2 models
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5.
# Assign the loaded detection model to global variable DET_MODEL
# TODO
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

cfg = get_cfg()

# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    )
)

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
    0.5  # set threshold for this model
)
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml.yaml"
)
DET_MODEL = DefaultPredictor(cfg)


def get_vehicle_coordinates(img):

    """
    This function will run an object detector over the the image, get
    the vehicle position in the picture and return it.

    Many things should be taken into account to make it work:
        1. Current model being used can detect up to 80 different objects,
           we're only looking for 'cars' or 'trucks', so you should ignore
           other detected objects.
        2. The object detector may find more than one vehicle in the picture,
           you must then, choose the one with the largest area in the image.
        3. The model can also fail and detect zero objects in the picture,
           in that case, you should return coordinates that cover the full
           image, i.e. [0, 0, width, height].
        4. Coordinates values must be integers, we're making reference to
           a position in a numpy.array, we can't use float values.

    Parameters
    ----------
    img : numpy.ndarray
        Image in RGB format.

    Returns
    -------
    box_coordinates : list
        List having bounding box coordinates as (left, top, right, bottom).
        Also known as (x1, y1, x2, y2).
   """
    # TODO
    outputs = DET_MODEL(img)
    mask_car_or_trucks = (outputs["instances"].pred_classes == 2) | (
        outputs["instances"].pred_classes == 7
    )
    bbox_car_trucks = outputs["instances"].pred_boxes[
        mask_car_or_trucks
    ]
    if bbox_car_trucks.tensor.size()[0] == 0:
        x1, y1, x2, y2 = (0, 0, img.shape[1], img.shape[0])
    else:
        index_max_area = bbox_car_trucks.area().argmax()
        x1, y1, x2, y2 = (
            bbox_car_trucks.tensor[index_max_area].to("cpu").numpy()
        )

    box_coordinates = int(x1), int(y1), int(x2), int(y2)

    return list(box_coordinates)
