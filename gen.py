import cv2
import numpy as np

from utils import build_chains2, get_instructions2
from cvoperations import center_images

# 读取关于如何拼这个图的指引
center, chains = get_instructions2("./instructions.json")
chains = build_chains2(center, chains)
center_idx = int(center.split(".")[0])

imgs = [str(key) + ".JPG" for key in chains.keys()]
imgs.append(center)
# imgs = os.listdir("./images/imgs_lesser")

detector = cv2.SIFT.create(contrastThreshold=0.07, edgeThreshold=2.5)
# detector = cv2.SIFT.create()
matcher = cv2.BFMatcher.create(crossCheck=True)

imgs = dict(
    [
        (int(img.split(".")[0]), cv2.imread("./images/imgs_lesser/" + img))
        for img in imgs
    ]
)

IMAGE_HEIGHT = imgs[center_idx].shape[0]
IMAGE_WIDTH = imgs[center_idx].shape[1]

# 扩展画布
CANVAS_HEIGHT, CANVAS_WIDTH = 4000, 6000

x_offset = (CANVAS_WIDTH - IMAGE_WIDTH) // 2
y_offset = (CANVAS_HEIGHT - IMAGE_HEIGHT) // 2
print(f"{x_offset=},{y_offset=}")

canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)

# cv2.imshow("canvas", cv2.resize(canvas, (1500, 1000)))
# cv2.waitKey(0)

# 把所有图像挪到画布中间
for img_idx in imgs.keys():
    canvas[y_offset : y_offset + IMAGE_HEIGHT, x_offset : x_offset + IMAGE_WIDTH, :] = (
        imgs[img_idx]
    )
    imgs[img_idx] = canvas.copy()
    cv2.imshow(f"{img_idx=}", cv2.resize(imgs[img_idx], (1500, 1000)))
    cv2.waitKey(0)
    cv2.destroyWindow(f"{img_idx=}")

# cv2.imshow("Canvas", cv2.resize(canvas, (1500, 1000)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 计算关键点
kps_descs_dict: dict[int, tuple[tuple[cv2.KeyPoint], np.ndarray]] = {}
for img_idx, img in imgs.items():
    if img_idx not in kps_descs_dict:
        kps_descs_dict[img_idx] = detector.detectAndCompute(img, None)

# 在透视链上进行关键点匹配，得到所有需要的DMatch对象
matches_dict: dict[tuple[int, int], tuple[cv2.DMatch]] = {}
for chain in chains.values():
    for junction in chain:
        if junction not in matches_dict:
            src_idx, dst_idx = junction
            # 注意是对所有描述符进行匹配，也就是上面标注的np.ndarray
            matches_dict[junction] = matcher.match(
                # 尤其注意这一行，是从query到train的匹配
                kps_descs_dict[src_idx][1],
                kps_descs_dict[dst_idx][1],
            )
            out = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH * 2, 3))
            kps1 = kps_descs_dict[src_idx][0]
            kps2 = kps_descs_dict[dst_idx][0]
            matches = matches_dict[junction]
            out = cv2.drawMatches(
                imgs[src_idx], kps1, imgs[dst_idx], kps2, matches, out, 10
            )
            # plt.imshow(out[:,:,::-1])
            # print(f"matches:{src_idx=},{dst_idx=}")
            # plt.show()
            # cv2.imshow(f"matches:{src_idx=},{dst_idx=}", out)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

# 构建所有需要的单应性矩阵
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

# 开拼
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
    canvas = cv2.copyTo(img, mask, canvas)

    # cv2.imshow(f"{img_idx=}", cv2.resize(img, (1500, 1000)))
    # cv2.waitKey(0)
    # cv2.destroyWindow(f"{img_idx=}")
    # plt.imshow(img[:,:,::-1])
    # plt.show()
cv2.imshow("Canvas", cv2.resize(canvas, (1500, 1000)))
cv2.waitKey(0)
cv2.destroyWindow("Canvas")
