import cv2


def center_images(
    imgs: dict[int, cv2.typing.MatLike],
    IMAGE_HEIGHT: int,
    IMAGE_WIDTH: int,
    x_offset: int,
    y_offset: int,
    canvas: cv2.typing.MatLike,
):
    for img_idx in imgs.keys():
        canvas[
            y_offset : y_offset + IMAGE_HEIGHT, x_offset : x_offset + IMAGE_WIDTH, :
        ] = imgs[img_idx]
        imgs[img_idx] = canvas.copy()
        cv2.imshow(f"{img_idx=}", cv2.resize(imgs[img_idx], (1500, 1000)))
        cv2.waitKey(0)
        cv2.destroyWindow(f"{img_idx=}")



