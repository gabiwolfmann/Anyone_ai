import utils
import os
import settings
from werkzeug.utils import secure_filename
from middleware import model_predict

from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    url_for,
    jsonify,
)

router = Blueprint("app_router", __name__, template_folder="templates")


@router.route("/", methods=["GET"])
def index():
    """
    Index endpoint, renders our HTML code.
    """
    return render_template("index.html")


@router.route("/", methods=["POST"])
def upload_image():
    """
    Function used in our frontend so we can upload and show an image.
    When it receives an image from the UI, it also calls our ML model to
    get and display the predictions.
    """
    # No file received, show basic UI
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    # File received but no filename is provided, show basic UI
    file = request.files["file"]
    if file.filename == "":
        flash("No image selected for uploading")
        return redirect(request.url)

    # File received and it's an image, we must show it and get predictions
    if file and utils.allowed_file(file.filename):
        # In order to correctly display the image in the UI and get model
        # predictions you should implement the following:
        #   1. Get an unique file name using utils.get_file_hash() function
        #    2. Store the image to disk using the new name
        #    3. Send the file to be processed by the `model` service
        #      Hint: Use middleware.model_predict() for sending jobs to model
        #            service using Redis.
        #    4. Update `context` dict with the corresponding values
        #  TODO
        hashed_filename = utils.get_file_hash(file)
        file.filename = hashed_filename
        filename_secure = secure_filename(file.filename)

        if not os.path.exists(
            os.path.join(settings.UPLOAD_FOLDER, filename_secure)
        ):
            file.save(os.path.join(settings.UPLOAD_FOLDER, filename_secure))

        predict, score = model_predict(file.filename)

        context = {
            "prediction": predict,
            "score": score * 100,  # to get the percentage score
            "filename": hashed_filename,
        }

        #  Update `render_template()` parameters as needed
        #  TODO
        return render_template(
            "index.html", filename=hashed_filename, context=context
        )
    # File received and but it isn't an image
    else:
        flash("Allowed image types are -> png, jpg, jpeg, gif")
        return redirect(request.url)


@router.route("/display/<filename>")
def display_image(filename):
    """
    Display uploaded image in our UI.
    """
    return redirect(
        url_for("static", filename="uploads/" + filename), code=301
    )


@router.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint used to get predictions without need to access the UI.

    Parameters
    ----------
    file : str
        Input image we want to get predictions from.

    Returns
    -------
    flask.Response
        JSON response from our API having the following format:
            {
                "success": bool,
                "prediction": str,
                "score": float,
            }

        - "success" will be True if the input file is valid and we get a
          prediction from our ML model.
        - "prediction" model predicted class as string.
        - "score" model confidence score for the predicted class as float.
    """
    # To correctly implement this endpoint you should:
    #    1. Check a file was sent and that file is an image
    #   2. Store the image to disk
    #   3. Send the file to be processed by the `model` service
    #      Hint: Use middleware.model_predict() for sending jobs to model
    #            service using Redis.
    #   4. Update and return `rpse` dict with the corresponding values
    # If user sends an invalid request (e.g. no file provided) this endpoint
    #  should return `rpse` dict with default values HTTP 400 Bad Request code
    #  TODO

    rpse = {"success": False, "prediction": None, "score": None}
    # No file received
    if "file" not in request.files:
        return jsonify(rpse), 400
    # File received but no filename is provided
    file = request.files["file"]
    if file.filename == "":
        return jsonify(rpse), 400

    # File received and it's an image, we must show it and get predictions
    if file and utils.allowed_file(file.filename):
        hashed_filename = utils.get_file_hash(file)
        file.filename = hashed_filename
        filename_secure = secure_filename(file.filename)

        if not os.path.exists(
            os.path.join(settings.UPLOAD_FOLDER, filename_secure)
        ):
            file.save(os.path.join(settings.UPLOAD_FOLDER, filename_secure))

        predict, score = model_predict(file.filename)

        rpse = {"success": True, "prediction": predict, "score": score}

        return jsonify(rpse), 200
    else:
        return jsonify(rpse), 400


@router.route("/feedback", methods=["GET", "POST"])
def feedback():
    """
    Store feedback from users about wrong predictions on a text file.

    Parameters
    ----------
    report : request.form
        Feedback given by the user with the following JSON format:
            {
                "filename": str,
                "prediction": str,
                "score": float
            }

        - "filename" corresponds to the image used stored in the uploads
          folder.
        - "prediction" is the model predicted class as string reported as
          incorrect.
        - "score" model confidence score for the predicted class as float.
    """
    #  Get reported predictions from `report` key
    report = request.form.get("report")

    # Store the reported data to a file on the corresponding path
    #  already provided in settings.py module
    #  TODO
    with open(settings.FEEDBACK_FILEPATH, "a") as f:
        f.write(str(report) + "\n")

    return render_template("index.html")
