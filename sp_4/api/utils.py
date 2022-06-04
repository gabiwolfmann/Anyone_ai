import os
import hashlib


def allowed_file(filename):
    """
    Checks if the format for the file received is acceptable. For this
    particular case, we must accept only image files.

    Parameters
    ----------
    filename : str
        Filename from werkzeug.datastructures.FileStorage file.

    Returns
    -------
    bool
        True if the file is an image, False otherwise.
    """
    #  Current implementation will allow any kind of file.
    # TODO -- done
    allowed_files = filename.lower().endswith(
        (".png", ".jpg", ".jpeg", ".gif")
    )
    return allowed_files


def get_file_hash(file):
    """
    Returns a new filename based on the file content using MD5 hashing.
    It uses hashlib.md5() function from Python standard library to get
    the hash.

    Parameters
    ----------
    file : werkzeug.datastructures.FileStorage
        File sent by user.

    Returns
    -------
    str
        New filename based in md5 file hash.
    """
    #  Current implementation will return the original file name.
    # TODO
    hashed_file = hashlib.md5(file.read()).hexdigest()
    file.seek(0)
    _, ext = os.path.splitext(file.filename)
    ext = ext.lower()
    hexa_filename = f"{hashed_file}{ext}"
    return hexa_filename
