import cv2
import numpy as np

from utils import build_chains2, get_instructions2
from cvoperations import (
    center_images,
    get_kps_descs_dict,
    get_matches_dict,
    get_warps_dict,
    stitch_images
)

def main():

    # Read the instructions on how to stitch the images.
    center, chains = get_instructions2("./instructions.json")
    chains = build_chains2(center, chains)
    center_idx = int(center.split(".")[0])

    imgs = [str(key) + ".JPG" for key in chains.keys()]
    imgs.append(center)

    detector = cv2.SIFT.create(contrastThreshold=0.07, edgeThreshold=2.5)
    matcher = cv2.BFMatcher.create(crossCheck=True)

    imgs = dict(
    [
        (int(img.split(".")[0]), cv2.imread("./images/imgs_lesser/" + img))
        for img in imgs
    ]
)

    IMAGE_HEIGHT = imgs[center_idx].shape[0]
    IMAGE_WIDTH = imgs[center_idx].shape[1]

    # Enlarge the canvas
    CANVAS_HEIGHT, CANVAS_WIDTH = 4000, 6000

    x_offset = (CANVAS_WIDTH - IMAGE_WIDTH) // 2
    y_offset = (CANVAS_HEIGHT - IMAGE_HEIGHT) // 2
    print(f"{x_offset=},{y_offset=}")

    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)

    # Center all the images to the larger canvas
    center_images(imgs, IMAGE_HEIGHT, IMAGE_WIDTH, x_offset, y_offset, canvas)

    # Calculate keypoints and get descriptors for each keypoint.
    kps_descs_dict = get_kps_descs_dict(imgs, detector)

    # Match all keypoints through the perspective chain.
    matches_dict = get_matches_dict(chains, kps_descs_dict, matcher)

    # Get all perspective transforming matrix
    warps_dict = get_warps_dict(chains, kps_descs_dict, matches_dict)

    # And stitch them all together
    stitch_images(chains, imgs, CANVAS_HEIGHT, CANVAS_WIDTH, canvas, warps_dict)

    cv2.imshow("Canvas", cv2.resize(canvas, (1500, 1000)))
    cv2.waitKey(0)
    cv2.destroyWindow("Canvas")

if __name__ == "__main__":
    main()
