import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class Detection:
    found: bool
    cx: float = 0.0
    cy: float = 0.0
    area: float = 0.0
    bbox: tuple = (0, 0, 0, 0)
    mask: np.ndarray = None


class BlueBlockTracker:
    """BGR 이미지에서 빨간색 블럭의 픽셀 중심을 추출한다.
    빨간색은 HSV H 채널이 0과 180 부근에서 두 구간으로 나뉘므로 두 범위를 OR 합산한다.
    """

    def __init__(self,
                 lower_hsv1=(0, 120, 50),
                 upper_hsv1=(10, 255, 255),
                 lower_hsv2=(170, 120, 50),
                 upper_hsv2=(180, 255, 255),
                 min_area=200,
                 morph_kernel=5):
        self.lower1 = np.array(lower_hsv1, dtype=np.uint8)
        self.upper1 = np.array(upper_hsv1, dtype=np.uint8)
        self.lower2 = np.array(lower_hsv2, dtype=np.uint8)
        self.upper2 = np.array(upper_hsv2, dtype=np.uint8)
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))

    def detect(self, bgr: np.ndarray) -> Detection:
        """BGR 이미지를 받아 가장 큰 빨간색 컨투어의 중심을 반환."""
        if bgr is None:
            return Detection(found=False)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.lower1, self.upper1),
            cv2.inRange(hsv, self.lower2, self.upper2),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Detection(found=False, mask=mask)

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.min_area:
            return Detection(found=False, area=area, mask=mask)

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return Detection(found=False, mask=mask)

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        x, y, w, h = cv2.boundingRect(largest)
        return Detection(found=True, cx=cx, cy=cy, area=area,
                         bbox=(x, y, w, h), mask=mask)
