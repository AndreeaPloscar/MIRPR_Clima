import pathlib
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_images(cold, warm, mixed, none):
    images = []
    classes = []
    for path in pathlib.Path(cold).iterdir():
        if path.is_file() and path.suffix == '.png':
            try:
                images.append(Image.open(path).convert('1'))
                classes.append(1)
            except Exception:
                print("could not open image")

    for path in pathlib.Path(warm).iterdir():
        if path.is_file() and path.suffix == '.png':
            try:
                images.append(Image.open(path).convert('1'))
                classes.append(2)
            except Exception:
                print("could not open image")

    for path in pathlib.Path(mixed).iterdir():
        if path.is_file() and path.suffix == '.png':
            try:
                images.append(Image.open(path).convert('1'))
                classes.append(3)
            except Exception:
                print("could not open image")

    for path in pathlib.Path(none).iterdir():
        if path.is_file() and path.suffix == '.png':
            try:
                images.append(Image.open(path).convert('1'))
                classes.append(0)
            except Exception:
                print("could not open image")

    return images, classes
