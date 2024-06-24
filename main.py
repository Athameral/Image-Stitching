import cv2
import numpy as np
import argparse

from stitcher import Stitcher


def main():
    stitcher = Stitcher(imageFolderPath="./images/imgs_lesser")
    stitcher.stitch()

    cv2.imwrite("./result.jpg", stitcher.canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imshow("Result", cv2.resize(stitcher.canvas, (1500, 1000)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
