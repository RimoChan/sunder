import numpy as np
import cv2


def 换(img, a1, a2, b1, b2):
    d = 2550
    a = np.array([
        [*a1, d],
        [*a2, d],
    ])
    b = np.array([
        b1,
        b2,
    ])

    x = np.linalg.lstsq(a, b, rcond=None)[0]

    # print('乘', x)
    # print('乘能', (x**2).mean())
    # print('验证', a@x)

    img = img.astype(np.float32)

    q = img[:, :, :1].copy()
    q[:] = d
    img = np.concatenate([img, q], axis=-1)

    img = img @ x

    img[np.where(img > 255)] = 255
    img[np.where(img < 0)] = 0

    return img.astype(np.uint8)


img = cv2.imread('o.png')


def 抠(img, color):
    mask = np.zeros(shape=img.shape, dtype=np.uint8)
    mask[np.where(img == color)] = 255
    mask = np.min(mask, axis=-1)
    return mask


def 超抠(img, color, th):
    mask = 抠(img, color)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    最大面积 = max([cv2.contourArea(cnt) for cnt in contours])
    好轮廓 = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 最大面积*th:
            好轮廓.append(cnt)
    mask = np.zeros(shape=mask.shape, dtype=np.uint8)
    cv2.drawContours(mask, 好轮廓, -1, (255, ), -1)
    return mask


def 分(img, color, th=0.03, d=5):
    mask = 超抠(img, color, th)
    a = 腐蚀(mask, d)
    b = 膨胀(mask, d)
    b[np.where(a == 255)] = 0
    return a, b


def 膨胀(img, d):
    kernel = np.ones((d, d), dtype=np.uint8)
    return cv2.dilate(img, kernel, 1)


def 腐蚀(img, d):
    kernel = np.ones((d, d), dtype=np.uint8)
    return cv2.erode(img, kernel, 1)


def sunder(img, 背景颜色, 我的背景颜色, 我的字颜色, 你的背景颜色, 你的字颜色):
    img = img.copy()
    img2 = 换(img, 背景颜色, 我的背景颜色, 背景颜色, 你的背景颜色)
    img3 = 换(img, 背景颜色, 你的背景颜色, 背景颜色, 我的背景颜色)
    img4 = 换(img, 我的背景颜色, 我的字颜色, 你的背景颜色, 你的字颜色)
    img5 = 换(img, 你的背景颜色, 你的字颜色, 我的背景颜色, 我的字颜色)

    a1, b1 = 分(img, 我的背景颜色)
    假面a1, 假面b1 = np.where(a1 == 255), np.where(b1 == 255)
    a2, b2 = 分(img, 你的背景颜色)
    假面a2, 假面b2 = np.where(a2 == 255), np.where(b2 == 255)

    img[假面a1] = img4[假面a1]
    img[假面a2] = img5[假面a2]

    img[假面b1] = img2[假面b1]
    img[假面b2] = img3[假面b2]

    return img
