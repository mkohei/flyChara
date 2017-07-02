# coding=utf-8

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation



##############################
########## CONSTANTS ##########
###############################





###############################
########## FUNCTIONS ##########
###############################





####################################
########## MAIN FUNCTIONS ##########
####################################
def main0():
    """ 射影変換で飛んでいく感じを表現できるか """
    # 画像の読み込み
    #img = cv2.imread('./data/krlab.png')
    img = cv2.imread('./data/google2.png')

    # 画像サイズの取得
    size = img.shape[:2][::-1]

    p1, p2, p3, p4 = [0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]
    #d1, d2, d3, d4 = [10, 10], [-10, 10], [0, -10], [0, -10]
    d1, d2, d3, d4 = [30, 30], [50, -50], [0, 0], [-30, -30]
    pst1 = np.float32([p1, p2, p3, p4])
    pst2 = np.float32([
        [p1[0]+d1[0], p1[1]+d1[1]], [p2[0]+d2[0], p2[1]+d2[1]], [p3[0]+d3[0], p3[1]+d3[1]], [p4[0]+d4[0], p4[1]+d4[1]]])

    # 透視変換行列を生成
    psp_matrix = cv2.getPerspectiveTransform(pst1, pst2)

    # 透視変換を行い、出力
    img_psp = cv2.warpPerspective(img, psp_matrix, size)

    # 表示
    cv2.imshow("origin", img)
    cv2.waitKey(0)
    cv2.imshow("Show PERSPECTIVE TRANSFORM Image", img_psp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return




##########################
########## MAIN ##########
##########################
if __name__ == '__main__':
    main0()
