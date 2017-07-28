# coding=utf-8

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation



##############################
########## CONSTANTS ##########
###############################
A4_H, A4_W = 210, 297




###############################
########## FUNCTIONS ##########
###############################





####################################
########## MAIN FUNCTIONS ##########
####################################
def test0():
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


def test1():
    """ 紙領域の抽出 """
    """ ４角の初期位置をあらかじめ決定しておき（角度と紙の大きさが変わらないなら一定のはず）
        その領域に紙の隅が来るように、紙を置いて処理を実行する (自動化を簡易化するため)
    """
    img = cv2.imread("./data/DSC_0014.JPG")
    origin = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    # コーナーと判定される座標(２次元配列のインデックス)を取得
    corner_points = np.where(dst > 0.01*dst.max())
    ypoints, xpoints = corner_points
    print(len(ypoints), len(xpoints), max(ypoints), max(xpoints))

    # 表示
    # ウィンドウのサイズを変更可能にする
    #cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.namedWindow("img")
    # マウスイベント時に関数 mouse_event の処理を行う
    points = []
    def mouse_event(event, x, y, flags, param):
        # クリック座標の表示
        if event == cv2.EVENT_LBUTTONUP:
            print(x,y)
            x_, y_ = None, None
            max_ = 1000000 # 半径100以上はリジェクト？
            # 一番近いコーナーを探索
            for i in range(len(ypoints)):
                d = (x-xpoints[i])**2 + (y-ypoints[i])**2
                # 更新
                if (d < max_):
                    x_, y_, max_ = xpoints[i], ypoints[i], d
            # 青い丸で印
            if x is None or y is None:
                print("Not Found")
            else:
                cv2.circle(img, (x_, y_), 20, (255, 0, 0), -1)
                #cv2.imshow("img", img)
                points.append([x_,y_])
                print(points)
    cv2.setMouseCallback("img", mouse_event)
    # show
    while (True):
        cv2.imshow("img", img)
        key = cv2.waitKey(1000)
        if key == 27:
            break
        if len(points) >= 4:
            break


    # 透視投影
    size = tuple(np.array([A4_W, A4_H]))
    per2 = np.float32([[0,0], [A4_W,0], [0,A4_H], [A4_W,A4_H]])
    per1 = np.float32(points)
    # 透視投影行列を生成
    psp_matrix = cv2.getPerspectiveTransform(per1, per2)
    # 透視変換を行い、出力
    img_psp = cv2.warpPerspective(origin, psp_matrix, size)
    cv2.imshow("out", img_psp)
    cv2.waitKey(0)



    cv2.destroyAllWindows()

    return


def test2():
    """ 文字領域の抽出 """
    """ test1() で得た４隅で透視投影によって紙を得, そこから文字領域の抽出を行う """
    img = cv2.imread("./data/DSC_0014.JPG")

    # 透視投影
    size = tuple(np.array([A4_W, A4_H]))
    per2 = np.float32([[0,0], [A4_W,0], [0,A4_H], [A4_W,A4_H]])
    per1 = np.float32([[1077, 751], [2179, 749], [872, 1088], [2409, 1082]])
    # 透視投影行列を生成
    psp_matrix = cv2.getPerspectiveTransform(per1, per2)
    # 透視変換を行い、出力
    img_psp = cv2.warpPerspective(img, psp_matrix, size)
    cv2.imshow("out", img_psp)

    # グレースケール
    gray = cv2.cvtColor(img_psp, cv2.COLOR_BGR2GRAY)
    string = gray.copy()
    string[string > 150] = 0
    cv2.imshow("gray", gray)
    cv2.imshow("string", string)

    plt.figure()
    #plt.hist((gray.flatten()), bins=64)
    plt.gray()
    plt.imshow(gray)
    plt.show()

    cv2.waitKey(0)


def test3():
    """ flyChara """
    ### const
    N = 3
    H, W = 210*N, 297*N # 変換後のサイズ
    th = 150 # 文字抽出時の閾値

    img = cv2.imread("../data/DSC_0014.JPG")
    imgH, imgW = img.shape[:2]

    ### 紙領域の抽出
    # ４角の検出(省略)
    corners = [[1077, 751], [2179, 749], [872, 1088], [2409, 1082]]
    # 射影変換
    per1 = np.float32(corners)
    per2 = np.float32([[0,0], [W,0], [0,H], [W,H]])
    mat12 = cv2.getPerspectiveTransform(per1, per2) # 1->2
    paper = cv2.warpPerspective(img, mat12, (W,H))

    ### 文字の抽出
    paper_gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    chars = paper.copy()
    chars[paper_gray > th] = 0
    ### 文字のインペイント
    paper_back = paper.copy()
    paper_ave = np.average(paper[paper_gray>th], axis=0)
    for i in range(3):
        paper_back[:, :, i] = paper_ave[i]

    ### 背景画像の作成(文字を動かすシーン画像)
    # 変換した紙背景を逆変換
    mat21 = cv2.getPerspectiveTransform(per2, per1) # 2->1
    paper_back_img = cv2.warpPerspective(paper_back, mat21, (imgW,imgH))
    # 背景シーンに挿入
    back = img.copy()
    for i in range(3):
        back_i, pbg_i = back[:, :, i], paper_back_img[:, :, i]
        back_i[pbg_i != 0] = pbg_i[pbg_i != 0]
        back[:, :, i] = back_i

    ### 文字の挿入
    # 良い感じに射影変換(動きの作成)
    points = [[1077, 751], [2179, 749], [872, 1088], [2409, 1082]]
    W = points[3][0] - points[2][0]
    H = W * 21.0 / 29.7
    move = True
    while move:
        move = False
        if points[0][0] > points[2][0]:
            points[0][0] -= 3
            move = True
        if points[1][0] > points[3][0]:
            points[2][0] += 3
            move = True
        if points[2][1]-points[0][1] < H:
            points[0][1] -= 3
        if points[3][1]-points[1][1] < H:
            points[1][1] -= 3

        per_c = np.float32(points)
        mat_c = cv2.getPerspectiveTransform(per2, per_c)
        chars_scene = cv2.warpPerspective(chars, mat_c, (imgW, imgH))
        # 背景シーンに挿入
        scene = back.copy()
        for i in range(3):
            scene_i, charsS_i = scene[:, :, i], chars_scene[:, :, i]
            scene_i[charsS_i != 0] = charsS_i[charsS_i != 0]
            scene[:, :, i] = scene_i


        #cv2.imshow("camera", img)
        #cv2.imshow("paper", paper)
        #cv2.imshow("chars", chars)
        #cv2.imshow("paper_back", paper_back)
        #cv2.imshow("paper_back_camera", paper_back_img)
        #cv2.imshow("back", back)
        #cv2.imshow("chars_scene", chars_scene)
        cv2.imshow("scene", scene)

        cv2.waitKey(0)


    

    


    return





##########################
########## MAIN ##########
##########################
if __name__ == '__main__':
    #test0()
    #test1()
    #test2()
    test3()
