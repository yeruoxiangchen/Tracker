import gzip
import struct
import json
from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
import imageio.v2 as iio
import png


def load_im(path: Union[str,Path]):
    """Loads an image from a file.

    :param path: Path to the image file to load.
    :return: ndarray with the loaded image.
    """
    im = iio.imread(path)
    return im

def save_im(path: Union[str,Path], im: npt.NDArray, jpg_quality: int =95):
    """Saves an image to a file.

    :param path: Path to the output image file.
    :param im: ndarray with the image to save.
    :param jpg_quality: Quality of the saved image (applies only to JPEG).
    """
    if Path(path).suffix.lower() in ["jpg", "jpeg"]:
        iio.imwrite(path, im, quality=jpg_quality)
    else:
        iio.imwrite(path, im, compression=3)


def load_depth(path: Union[str,Path]):
    """Loads a depth image from a file.

    :param path: Path to the depth image file to load.
    :return: ndarray with the loaded depth image.
    """
    return iio.imread(path).astype(np.float32)


def save_depth(path: Union[str,Path], im: npt.NDArray):
    """Saves a depth image (16-bit) to a PNG file.

    :param path: Path to the output depth image file.
    :param im: ndarray with the depth image to save.
    """
    if Path(path).suffix.lower() != ".png":
        raise ValueError("Only PNG format is currently supported.")

    im_uint16 = np.round(im).astype(np.uint16)

    # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
    w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
    with open(path, "wb") as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))


def load_json(path: Union[str,Path], keys_to_int=False):
    """Loads content of a JSON file.

    :param path: Path to the JSON file. If ".json.gz" extension, opens with gzip.
    :return: Content of the loaded JSON file.
    """
    path = Path(path)
    assert path.as_posix().endswith(('.json', '.json.gz')), f"{path} should end with .json or .json.gz extension"

    # Keys to integers.
    def convert_keys_to_int(x):
        return {int(k) if k.lstrip("-").isdigit() else k: v for k, v in x.items()}
    
    # Open+decompress with gzip if ".json.gz" file extension
    if path.as_posix().endswith('.json.gz'):
        f = gzip.open(path, "rt", encoding="utf8")
    else:
        f = open(path, "r")
    if keys_to_int:
        content = json.load(f, object_hook=lambda x: convert_keys_to_int(x))
    else:
        content = json.load(f)

    f.close()

    return content


def save_json(path: Union[str,Path], content: dict, compress=False, verbose=False):
    """Saves the provided content to a JSON file.

    :param path: Path to the output JSON file.
    :param content: Dictionary/list to save.
    :param compress: Saves as a gzip archive, appends ".gz" extension to filepath.
    """
    path = Path(path)
    assert path.as_posix().endswith(('.json', '.json.gz')), f"{path} should end with .json or .json.gz extension"
    
    if compress:
        if path.suffix == '.json':
            path = path.parent / (path.stem + ".json.gz")
        f = gzip.open(path, "wt", encoding="utf8")
    else:
        f = open(path, "w")

    if isinstance(content, dict):
        f.write("{\n")
        content_sorted = sorted(content.items(), key=lambda x: x[0])
        for elem_id, (k, v) in enumerate(content_sorted):
            f.write('  "{}": {}'.format(k, json.dumps(v, sort_keys=True)))
            if elem_id != len(content) - 1:
                f.write(",")
            f.write("\n")
        f.write("}")

    elif isinstance(content, list):
        f.write("[\n")
        for elem_id, elem in enumerate(content):
            f.write("  {}".format(json.dumps(elem, sort_keys=True)))
            if elem_id != len(content) - 1:
                f.write(",")
            f.write("\n")
        f.write("]")

    else:
        json.dump(content, f, sort_keys=True)

    f.close()