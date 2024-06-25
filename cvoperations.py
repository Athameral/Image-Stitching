import cv2
import numpy as np

from tqdm import tqdm

def center_images(
    imgs: dict[str, cv2.typing.MatLike],
    IMAGE_HEIGHT: int,
    IMAGE_WIDTH: int,
    x_offset: int,
    y_offset: int,
    canvas: cv2.typing.MatLike,
):
    for img_name in tqdm(imgs.keys(), desc="Centering Images"):
        canvas[
            y_offset : y_offset + IMAGE_HEIGHT, x_offset : x_offset + IMAGE_WIDTH, :
        ] = imgs[img_name]
        imgs[img_name] = canvas.copy()
        # cv2.imshow(f"{img_idx=}", cv2.resize(imgs[img_idx], (1500, 1000)))
        # cv2.waitKey(0)
        # cv2.destroyWindow(f"{img_idx=}")


def get_kps_descs_dict(imgs: dict[str, cv2.typing.MatLike], detector: cv2.SIFT):
    kps_descs_dict: dict[str, tuple[tuple[cv2.KeyPoint], np.ndarray]] = {}
    for img_name, img in tqdm(imgs.items(), desc="Running SIFT"):
        if img_name not in kps_descs_dict:
            kps_descs_dict[img_name] = detector.detectAndCompute(img, None)
    return kps_descs_dict


def get_matches_dict(
    chains: dict[str, list[tuple[str, str]]],
    kps_descs_dict: dict[str, tuple[tuple[cv2.KeyPoint], np.ndarray]],
    matcher: cv2.BFMatcher,
):
    matches_dict: dict[tuple[str, str], tuple[cv2.DMatch]] = {}
    for chain in tqdm(chains.values(), desc="Matching"):
        for junction in chain:
            if junction not in matches_dict:
                src_name, dst_name = junction
                # Note that, we try to match the descriptors here, i.e.,
                # calculate the distance between every 2 descriptors.
                matches_dict[junction] = matcher.match(
                    # Take especial care for this, query is the source,
                    # whereas train is the destination.
                    kps_descs_dict[src_name][1],
                    kps_descs_dict[dst_name][1],
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
    chains: dict[str, list[tuple[str, str]]],
    kps_descs_dict: dict[str, tuple[tuple[cv2.KeyPoint], np.ndarray]],
    matches_dict: dict[tuple[str, str], tuple[cv2.DMatch]],
):
    warps_dict: dict[tuple[str, str], np.ndarray] = {}
    for chain in tqdm(chains.values(), desc="Calculating Homography"):
        for junction in chain:
            if junction not in warps_dict:
                src_name, dst_name = junction
                matches = matches_dict[junction]
                P, mask = cv2.findHomography(
                    np.array(
                        [kps_descs_dict[src_name][0][m.queryIdx].pt for m in matches]
                    ).reshape(-1, 2),
                    np.array(
                        [kps_descs_dict[dst_name][0][m.trainIdx].pt for m in matches]
                    ).reshape(-1, 2),
                    cv2.RANSAC,
                )

                # if dst_idx == center_idx:
                #     P[0, 2] += x_offset
                #     P[1, 2] += y_offset

                warps_dict[junction] = P
                # print(f"P{src_name=},{dst_name=}=\n{P}")
    return warps_dict


def stitch_images(
    chains: dict[str, list[tuple[str, str]]],
    imgs: dict[str, cv2.typing.MatLike],
    CANVAS_HEIGHT: int,
    CANVAS_WIDTH: int,
    canvas: cv2.typing.MatLike,
    warps_dict: dict[tuple[str, str], np.ndarray],
):
    for img_name, chain in tqdm(chains.items(), "Stitching"):
        img = imgs[img_name]
        for junction in chain:
            P = warps_dict[junction]
            img = cv2.warpPerspective(
                src=img,
                M=P,
                # In OpenCV, widths are assigned first, while heights second.
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
