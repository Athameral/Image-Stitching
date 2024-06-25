import cv2
import numpy as np
import argparse

from stitcher import Stitcher

parser = argparse.ArgumentParser()
parser.add_argument("--contrastThreshold", type=float, default=0.07)
parser.add_argument("--edgeThreshold", type=float, default=2.5)
parser.add_argument("instructions", type=str, help="path to instructions.json")
parser.add_argument("imagesFolder", type=str, help="path to the images folder")

args = parser.parse_args().__dict__

def main():
    stitcher = Stitcher(
        contrastThreshold=args["contrastThreshold"],
        edgeThreshold=args["edgeThreshold"],
        instructionPath=["instructions"],
        imageFolderPath=args["imagesFolder"],
    )
    stitcher.stitch()

    cv2.imwrite("./result.jpg", stitcher.canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imshow("Result", cv2.resize(stitcher.canvas, (1500, 1000)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
