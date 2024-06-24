import cv2
import numpy as np
import os

from utils import build_chains2, get_instructions2
from cvoperations import (
    center_images,
    get_kps_descs_dict,
    get_matches_dict,
    get_warps_dict,
    stitch_images,
)


class Stitcher:

    def __init__(
        self,
        contrastThreshold: float = 0.07,
        edgeThreshold: float = 2.5,
        instructionPath: str = "./instructions.json",
        imageFolderPath: str = "./images",
        CANVAS_HEIGHT: int = 4000,
        CANVAS_WIDTH: int = 6000,
    ) -> None:
        self.detector = cv2.SIFT.create(
            contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold
        )
        self.matcher = cv2.BFMatcher.create(crossCheck=True)
        self.instructionPath = instructionPath
        self.imageFolderPath = imageFolderPath
        self.CANVAS_HEIGHT = CANVAS_HEIGHT
        self.CANVAS_WIDTH = CANVAS_WIDTH
        self.canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)

    def stitch(self, showResult: bool = False, savePath: str | None = None):
        # Read the instructions on how to stitch the images.
        center, chains = get_instructions2("./instructions.json")
        chains = build_chains2(center, chains)

        imgs = list(chains.keys())
        imgs = dict(
            [(img, cv2.imread(os.path.join(self.imageFolderPath, img))) for img in imgs]
        )

        IMAGE_HEIGHT = imgs[center].shape[0]
        IMAGE_WIDTH = imgs[center].shape[1]

        # Enlarge the canvas
        x_offset = (self.CANVAS_WIDTH - IMAGE_WIDTH) // 2
        y_offset = (self.CANVAS_HEIGHT - IMAGE_HEIGHT) // 2

        # Center all the images to the larger canvas
        center_images(imgs, IMAGE_HEIGHT, IMAGE_WIDTH, x_offset, y_offset, self.canvas)

        # Calculate keypoints and get descriptors for each keypoint.
        kps_descs_dict = get_kps_descs_dict(imgs, self.detector)

        # Match all keypoints through the perspective chain.
        matches_dict = get_matches_dict(chains, kps_descs_dict, self.matcher)

        # Get all perspective transforming matrix
        warps_dict = get_warps_dict(chains, kps_descs_dict, matches_dict)

        # And stitch them all together
        stitch_images(
            chains, imgs, self.CANVAS_HEIGHT, self.CANVAS_WIDTH, self.canvas, warps_dict
        )
