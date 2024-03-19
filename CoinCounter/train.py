from lib.detector import pretreat, segment_with_watershed
from lib.utils import read_images
import cv2


def train(images):
    for image in images:
        res = pretreat(image)
        res, num_regions = segment_with_watershed(res)
        print(num_regions)


if __name__ == "__main__":
    image_dataset = read_images()
    image_list = list(image_dataset.values())[:20]
    train(image_list)
