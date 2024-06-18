import cv2
import numpy as np


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
        # cv2.imshow(f"{img_idx=}", cv2.resize(imgs[img_idx], (1500, 1000)))
        # cv2.waitKey(0)
        # cv2.destroyWindow(f"{img_idx=}")


def get_kps_descs_dict(imgs: dict[int, cv2.typing.MatLike], detector: cv2.SIFT):
    kps_descs_dict: dict[int, tuple[tuple[cv2.KeyPoint], np.ndarray]] = {}
    for img_idx, img in imgs.items():
        if img_idx not in kps_descs_dict:
            kps_descs_dict[img_idx] = detector.detectAndCompute(img, None)
    return kps_descs_dict


def get_matches_dict(
    chains: dict[int, list[tuple[int, int]]],
    kps_descs_dict: dict[int, tuple[tuple[cv2.KeyPoint], np.ndarray]],
    matcher: cv2.BFMatcher,
):
    matches_dict: dict[tuple[int, int], tuple[cv2.DMatch]] = {}
    for chain in chains.values():
        for junction in chain:
            if junction not in matches_dict:
                src_idx, dst_idx = junction
                # Note that, we try to match the descriptors here, i.e.,
                # calculate the distance between every 2 descriptors.
                matches_dict[junction] = matcher.match(
                    # Take especial care for this, query is the source,
                    # whereas train is the destination.
                    kps_descs_dict[src_idx][1],
                    kps_descs_dict[dst_idx][1],
                )
    # out = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH * 2, 3))
    # kps1 = kps_descs_dict[src_idx][0]
    # kps2 = kps_descs_dict[dst_idx][0]
    # matches = matches_dict[junction]
    # out = cv2.drawMatches(
    #     imgs[src_idx], kps1, imgs[dst_idx], kps2, matches, out, 10
    # )
    return matches_dict


def get_warps_dict(
    chains: dict[int, list[tuple[int, int]]],
    kps_descs_dict: dict[int, tuple[tuple[cv2.KeyPoint], np.ndarray]],
    matches_dict: dict[tuple[int, int], tuple[cv2.DMatch]],
):
    warps_dict: dict[tuple[int, int], np.ndarray] = {}
    for chain in chains.values():
        for junction in chain:
            if junction not in warps_dict:
                src_idx, dst_idx = junction
                matches = matches_dict[junction]
                P, mask = cv2.findHomography(
                    np.array(
                        [kps_descs_dict[src_idx][0][m.queryIdx].pt for m in matches]
                    ).reshape(-1, 2),
                    np.array(
                        [kps_descs_dict[dst_idx][0][m.trainIdx].pt for m in matches]
                    ).reshape(-1, 2),
                    cv2.RANSAC,
                )

                # if dst_idx == center_idx:
                #     P[0, 2] += x_offset
                #     P[1, 2] += y_offset

                warps_dict[junction] = P
                print(f"P{src_idx=},{dst_idx=}=\n{P}")
    return warps_dict


def stitch_images(
    chains: dict[int, list[tuple[int, int]]],
    imgs: dict[int, cv2.typing.MatLike],
    CANVAS_HEIGHT: int,
    CANVAS_WIDTH: int,
    canvas: cv2.typing.MatLike,
    warps_dict: dict[tuple[int, int], np.ndarray],
):
    for img_idx, chain in chains.items():
        img = imgs[img_idx]
        for junction in chain:
            P = warps_dict[junction]
            img = cv2.warpPerspective(
                src=img,
                M=P,
                # 在OpenCV中，图像size一般是先指定宽，后指定高
                dsize=(CANVAS_WIDTH, CANVAS_HEIGHT),
                dst=None,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        mask = (img > 0).astype(np.uint8) * 255
        # cv2.imshow(f"mask: {img_idx=}", cv2.resize(mask, (1500, 1000)))

        # Code below will give awful result, it's not correct.
        # canvas = cv2.seamlessClone(img, canvas, mask, (CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2), cv2.NORMAL_CLONE)
        # canvas = cv2.copyTo(img, mask, canvas)
        cv2.copyTo(img, mask, canvas)
