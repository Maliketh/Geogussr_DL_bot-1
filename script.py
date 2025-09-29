import os
from PIL import Image


def crop_and_overwrite_images(root_dir, crop_left=0.2, crop_right=0.2):
    """
    Recursively crops all JPEG images in a directory.
    Keeps filenames and folder structure the same, overwriting files in place.

    :param root_dir: Root dataset directory containing images.
    :param crop_left: Fraction of width to crop from left side.
    :param crop_right: Fraction of width to crop from right side.
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):
                file_path = os.path.join(subdir, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        left = int(width * crop_left)
                        right = int(width * (1 - crop_right))
                        top, bottom = 0, height

                        cropped = img.crop((left, top, right, bottom))
                        cropped.save(file_path, quality=95)  # overwrite in place
                        print(f"Cropped: {file_path}")
                except Exception as e:
                    print(f"Failed on {file_path}: {e}")


# insert your dataset directory here
dataset_dir = r"an_example/dataset"
crop_and_overwrite_images(dataset_dir, crop_left=0.2, crop_right=0.2)
