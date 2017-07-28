# encoding=utf-8

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------------------
# ---------- CONSTANTS ----------
# -------------------------------
CORNER = [[666,458], [1347,444], [580,771], [1462,758]]

RED, GREEN, BLUE = (0, 0, 255), (0, 255, 0), (255,0, 0)

R = 50



# -------------------------------
# ---------- VARIABLES ----------
# -------------------------------
corner = []


# ------------------------------------
# ---------- MAIN FUNCTIONS ----------
# ------------------------------------
def test0():
    cap = cv2.VideoCapture(1)
    while True:
        corner = []
        ret, frame = cap.read()

        def mouse_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONUP:
                print(x, y)

        # コーナー検出
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        corner_points = np.where(dst > 0.01*dst.max())
        ypoints, xpoints = corner_points

        for p in CORNER:
            d = (p[0]-xpoints)**2 + (p[1]-ypoints)**2
            i = np.argmin(d)
            corner.append((xpoints[i], ypoints[i]))
            # 検出コーナー
            cv2.circle(frame, (xpoints[i], ypoints[i]), 10, BLUE, -1)
            # 定義コーナー
            cv2.circle(frame, (p[0], p[1]), 50, RED, 5)
        # 検出矩形
        print(corner)
        if len(corner) > 3:
            cv2.line(frame, (corner[0][0],corner[0][1]), (corner[1][0],corner[1][1]), BLUE, 3)
            cv2.line(frame, (corner[0][0],corner[0][1]), (corner[2][0],corner[2][1]), BLUE, 3)
            cv2.line(frame, (corner[1][0],corner[1][1]), (corner[3][0],corner[3][1]), BLUE, 3)
            cv2.line(frame, (corner[2][0],corner[2][1]), (corner[3][0],corner[3][1]), BLUE, 3)
        
        cv2.imshow('cam', frame)
        key = cv2.waitKey(0)
        # Enter
        if key == 13:
            break


# --------------------------
# ---------- MAIN ----------
# --------------------------
if __name__ == '__main__':
    test0()
