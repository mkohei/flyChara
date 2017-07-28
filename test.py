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

    img = cv2.imread("./data/DSC_0014.JPG")
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
    per_c = np.float32([[872, 500], [2409, 500], [872, 1088], [2409, 1088]])
    mat_c = cv2.getPerspectiveTransform(per2, per_c)
    chars_scene = cv2.warpPerspective(chars, mat_c, (imgW, imgH))
    # 背景シーンに挿入
    scene = back.copy()
    for i in range(3):
        scene_i, charsS_i = scene[:, :, i], chars_scene[:, :, i]
        scene_i[charsS_i != 0] = charsS_i[charsS_i != 0]
        scene[:, :, i] = scene_i


    cv2.imshow("camera", img)
    #cv2.imshow("paper", paper)
    #cv2.imshow("chars", chars)
    #cv2.imshow("paper_back", paper_back)
    #cv2.imshow("paper_back_camera", paper_back_img)
    #cv2.imshow("back", back)
    #cv2.imshow("chars_scene", chars_scene)
    cv2.imshow("scene", scene)

    cv2.waitKey(0)







    return

def test4():
    global per2, chars, back, imgW, imgH
    """ flyChara """
    ### const
    N = 3
    H, W = 210*N, 297*N # 変換後のサイズ
    th = 150 # 文字抽出時の閾値

    img = cv2.imread("./data/DSC_0014.JPG")
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

    # 紙の辺の長さ
    cornerUpW = corners[1][0] - corners[0][0]    #上辺の幅
    cornerDownW = corners[3][0] - corners[2][0]  #下辺の幅
    cornerLeftH = corners[2][1] - corners[0][1]  #左辺の高さ
    cornerRightH = corners[3][1] - corners[1][1] #右辺の高さ

    beforeImg = corners #動かす前の画像をコピー
    #-----左から右に縮む動き-----
    upRight = [corners[0][0]+cornerUpW*0.7 , corners[0][1]+cornerLeftH*0.1]
    upLeft =  [corners[1][0] , corners[1][1]]
    downRight = [corners[2][0]+cornerDownW*0.7 , corners[2][1]-cornerRightH*0.1]
    downLeft = [corners[3][0] , corners[3][1]]
    movePoint = np.float32([upRight, upLeft, downRight, downLeft]) #移動先の座標指定
    textmove(movePoint, beforeImg, 40) #移動先座標，移動前座標，フレーム数
    beforeImg = movePoint #座標の更新
    #-----右から左に伸びる動き-----
    upRight = [0 , corners[0][1]-cornerLeftH*0.1]
    upLeft =  [corners[1][0] , corners[1][1]]
    downRight = [0 , corners[2][1]+cornerLeftH*0.1]
    downLeft = [corners[3][0] , corners[3][1]]
    movePoint = np.float32([upRight, upLeft, downRight, downLeft])
    textmove(movePoint, beforeImg, 40) #移動先座標，移動前座標，フレーム数
    beforeImg = movePoint #座標の更新
    #-----左から右に反転する動き-----
    upRight = [corners[0][0] , corners[0][1]-cornerLeftH*0.3]
    upLeft =  [0 , corners[1][1]]
    downRight = [corners[2][0] , corners[2][1]+cornerLeftH*0.3]
    downLeft = [0, corners[3][1]]
    movePoint = np.float32([upRight, upLeft, downRight, downLeft])
    textmove(movePoint, beforeImg, 40) #移動先座標，移動前座標，フレーム数
    beforeImg = movePoint #座標の更新
    #-----画面にへばりつく動き-----
    upRight = [0, 0]
    upLeft =  [imgW , 0]
    downRight = [imgW*0.2 , imgH]
    downLeft = [imgW*0.8, corners[3][1]]
    movePoint = np.float32([upRight, upLeft, downRight, downLeft])
    textmove(movePoint, beforeImg, 40) #移動先座標，移動前座標，フレーム数
    beforeImg = movePoint #座標の更新

    cv2.waitKey(0)
    return


def textmove(movePoint, beforeImg, fream):#移動先座標，移動前座標，フレーム数
    mv = movePoint - beforeImg
    oneMv = mv/fream
    inpkt = 0
    inpktPoint = 0
    inpktMv = 0
    oneInpktMv = 0
    if (beforeImg[0][0] <= beforeImg[1][0] and movePoint[0][0] >= movePoint[1][0] or\
    beforeImg[2][0] <= beforeImg[3][0] and movePoint[2][0] >= movePoint[3][0]  ):
        for f in range(fream):
            if ( beforeImg[0][0]+(oneMv[0][0]*f) >= beforeImg[1][0]+(oneMv[1][0]*f) ):
                inpkt = f
                inp = (beforeImg[1][0] + beforeImg[0][0])/2
                inpktPoint = np.float32([ [inp,movePoint[0][1]],[inp,movePoint[1][1]],\
                [inp,movePoint[2][1]],[inp,movePoint[3][1]] ])
                break
            if (beforeImg[2][0]+(oneMv[2][0]*f) >= beforeImg[3][0]+(oneMv[3][0]*f) ):
                inpkt = f
                inp = (beforeImg[2][0] + beforeImg[3][0])/2
                inpktPoint = np.float32([ [inp,movePoint[0][1]],[inp,movePoint[1][1]],\
                [inp,movePoint[2][1]],[inp,movePoint[3][1]] ])
                break
    elif(beforeImg[0][0] >= beforeImg[1][0] and movePoint[0][0] <= movePoint[1][0] or\
    beforeImg[2][0] >= beforeImg[3][0] and movePoint[2][0] <= movePoint[3][0]  ):
        for f in range(fream):
            if ( beforeImg[0][0]+(oneMv[0][0]*f) <= beforeImg[1][0]+(oneMv[1][0]*f) ):
                inpkt = f
                inp = (beforeImg[1][0] + beforeImg[0][0])/2
                inpktPoint = np.float32([ [inp,movePoint[0][1]],[inp,movePoint[1][1]],\
                [inp,movePoint[2][1]],[inp,movePoint[3][1]] ])
                break
            if (beforeImg[2][0]+(oneMv[2][0]*f) <= beforeImg[3][0]+(oneMv[3][0]*f) ):
                inpkt = f
                inp = (beforeImg[2][0] + beforeImg[3][0])/2
                inpktPoint = np.float32([ [inp,movePoint[0][1]],[inp,movePoint[1][1]],\
                [inp,movePoint[2][1]],[inp,movePoint[3][1]] ])
                break
    if (inpkt != 0):
        mv = inpktPoint - beforeImg
        oneMv = mv/inpkt

    for f in range(fream):
        if (f == inpkt+1 and f !=0):
            beforeImg = per_c
            mv = movePoint - per_c
            oneMv = mv/(fream-inpkt)
        if (f <= inpkt and inpkt != 0):
            per_c = np.float32(beforeImg + oneMv*(f))
        else:
            per_c = np.float32(beforeImg + oneMv*(f-inpkt))

        if ( (per_c[0][0]<=per_c[1][0]+10 and per_c[2][0]>=per_c[3][0]-10) or (per_c[0][0]>=per_c[1][0] and per_c[2][0]-10<=per_c[3][0]+10) ):
            cv2.imshow("scene", back)
            continue

        if ( (per_c[0][1]<=per_c[2][1] and per_c[1][1]>=per_c[3][1]) or (per_c[0][1]>=per_c[2][1] and per_c[1][1]<=per_c[3][1]) ):
            cv2.imshow("scene", back)
            continue

        mat_c = cv2.getPerspectiveTransform(per2, per_c)
        chars_scene = cv2.warpPerspective(chars, mat_c, (imgW, imgH))
        # 背景シーンに挿入
        scene = back.copy()
        for i in range(3):
            scene_i, charsS_i = scene[:, :, i], chars_scene[:, :, i]
            scene_i[charsS_i != 0] = charsS_i[charsS_i != 0]
            scene[:, :, i] = scene_i
        cv2.imshow("scene", scene)
        cv2.waitKey(1)

    return

    inpkt = [-1,-1,-1,-1]
    inpktPoint = [[],[],[],[]]
    #global per2, chars, back, imgW, imgH
    #per2, chars, back, imgW, imgHはグローバルで指定をお願いします。
    def animeFream(corners, getFream):
        global inpkt
        global inpktPoint
        cornerUpW = corners[1][0] - corners[0][0]    #上辺の幅
        cornerDownW = corners[3][0] - corners[2][0]  #下辺の幅
        cornerLeftH = corners[2][1] - corners[0][1]  #左辺の高さ
        cornerRightH = corners[3][1] - corners[1][1] #右辺の高さ
        freamSet = [40,80,120,160]
        if getFream < freamSet[0]:
            #-----左から右に縮む動き-----
            BupRight = [corners[0][0] , corners[0][1]]
            BupLeft =  [corners[1][0] , corners[1][1]]
            BdownRight = [corners[2][0] , corners[2][1]]
            BdownLeft = [corners[3][0] , corners[3][1]]
            AupRight = [corners[0][0]+cornerUpW*0.7 , corners[0][1]+cornerLeftH*0.1]
            AupLeft =  [corners[1][0] , corners[1][1]]
            AdownRight = [corners[2][0]+cornerDownW*0.7 , corners[2][1]-cornerRightH*0.1]
            AdownLeft = [corners[3][0] , corners[3][1]]
            moveSelect = 0;
            moveFream = freamSet[0];
        elif getFream < freamSet[1] and getFream >= freamSet[0]:
            #-----右から左に伸びる動き-----
            BupRight = [corners[0][0]+cornerUpW*0.7 , corners[0][1]+cornerLeftH*0.1]
            BupLeft =  [corners[1][0] , corners[1][1]]
            BdownRight = [corners[2][0]+cornerDownW*0.7 , corners[2][1]-cornerRightH*0.1]
            BdownLeft = [corners[3][0] , corners[3][1]]
            AupRight = [0 , corners[0][1]-cornerLeftH*0.1]
            AupLeft =  [corners[1][0] , corners[1][1]]
            AdownRight = [0 , corners[2][1]+cornerLeftH*0.1]
            AdownLeft = [corners[3][0] , corners[3][1]]
            moveSelect = 1;
            moveFream = freamSet[1] - freamSet[0];
        elif getFream < freamSet[2] and getFream >= freamSet[1]:
            #-----左から右に反転する動き-----
            BupRight = [0 , corners[0][1]-cornerLeftH*0.1]
            BupLeft =  [corners[1][0] , corners[1][1]]
            BdownRight = [0 , corners[2][1]+cornerLeftH*0.1]
            BdownLeft = [corners[3][0] , corners[3][1]]
            AupRight = [corners[0][0] , corners[0][1]-cornerLeftH*0.3]
            AupLeft =  [0 , corners[1][1]]
            AdownRight = [corners[2][0] , corners[2][1]+cornerLeftH*0.3]
            AdownLeft = [0, corners[3][1]]
            moveSelect = 2;
            moveFream = freamSet[2] - freamSet[1];
        elif getFream < freamSet[3] and getFream >= freamSet[2]:
            #-----画面にへばりつく動き-----
            BupRight = [corners[0][0] , corners[0][1]-cornerLeftH*0.3]
            BupLeft =  [0 , corners[1][1]]
            BdownRight = [corners[2][0] , corners[2][1]+cornerLeftH*0.3]
            BdownLeft = [0, corners[3][1]]
            AupRight = [0, 0]
            AupLeft =  [imgW , 0]
            AdownRight = [imgW*0.2 , imgH]
            AdownLeft = [imgW*0.8, corners[3][1]]
            moveSelect = 3;
            moveFream = freamSet[3] - freamSet[2];


        beforeImg = np.float32([BupRight, BupLeft, BdownRight, BdownLeft])
        movePoint = np.float32([AupRight, AupLeft, AdownRight, AdownLeft])
        mv = movePoint - beforeImg
        oneMv = mv/moveFream
        if (inpkt[moveSelect] == -1):
            if (beforeImg[0][0] <= beforeImg[1][0] and movePoint[0][0] >= movePoint[1][0] or\
            beforeImg[2][0] <= beforeImg[3][0] and movePoint[2][0] >= movePoint[3][0]  ):
                for f in range(moveFream):
                    if ( beforeImg[0][0]+(oneMv[0][0]*f) >= beforeImg[1][0]+(oneMv[1][0]*f) ):
                        inpkt[moveSelect]  = f
                        inp = (beforeImg[1][0] + beforeImg[0][0])/2
                        inpktPoint[moveSelect] = np.float32([ [inp,movePoint[0][1]],[inp,movePoint[1][1]],\
                        [inp,movePoint[2][1]],[inp,movePoint[3][1]] ])
                        break
                    if (beforeImg[2][0]+(oneMv[2][0]*f) >= beforeImg[3][0]+(oneMv[3][0]*f) ):
                        inpkt[moveSelect]  = f
                        inp = (beforeImg[2][0] + beforeImg[3][0])/2
                        inpktPoint[moveSelect] = np.float32([ [inp,movePoint[0][1]],[inp,movePoint[1][1]],\
                        [inp,movePoint[2][1]],[inp,movePoint[3][1]] ])
                        break
                if (inpkt[moveSelect] == -1):
                    inpkt[moveSelect] = 0
            elif(beforeImg[0][0] >= beforeImg[1][0] and movePoint[0][0] <= movePoint[1][0] or\
            beforeImg[2][0] >= beforeImg[3][0] and movePoint[2][0] <= movePoint[3][0]  ):
                for f in range(moveFream):
                    if ( beforeImg[0][0]+(oneMv[0][0]*f) <= beforeImg[1][0]+(oneMv[1][0]*f) ):
                        inpkt[moveSelect]  = f
                        inp = (beforeImg[1][0] + beforeImg[0][0])/2
                        inpktPoint[moveSelect] = np.float32([ [inp,movePoint[0][1]],[inp,movePoint[1][1]],\
                        [inp,movePoint[2][1]],[inp,movePoint[3][1]] ])
                        break
                    if (beforeImg[2][0]+(oneMv[2][0]*f) <= beforeImg[3][0]+(oneMv[3][0]*f) ):
                        inpkt[moveSelect]  = f
                        inp = (beforeImg[2][0] + beforeImg[3][0])/2
                        inpktPoint[moveSelect] = np.float32([ [inp,movePoint[0][1]],[inp,movePoint[1][1]],\
                        [inp,movePoint[2][1]],[inp,movePoint[3][1]] ])
                        break
                if (inpkt[moveSelect] == -1):
                    inpkt[moveSelect] = 0
            else:
                inpkt[moveSelect] = 0
        blockFream = freamSet[moveSelect] - moveFream
        if (inpkt[moveSelect] != 0):
            #inp = inpktPoint[moveSelect, :, :]
            inst = inpktPoint[moveSelect]
            mv = inst - beforeImg
            oneMv = mv/inpkt[moveSelect]
        if (getFream-blockFream > inpkt[moveSelect] and inpkt[moveSelect] != 0):
            beforeImg = np.float32(beforeImg + oneMv*(getFream-blockFream))
            mv = movePoint - beforeImg
            oneMv = mv/(moveFream-inpkt[moveSelect])
        if (getFream-blockFream <= inpkt[moveSelect] and inpkt[moveSelect] != 0):
            per_c = np.float32(beforeImg + oneMv*(getFream-blockFream))
        else:
            per_c = np.float32(beforeImg + oneMv*(getFream-blockFream-inpkt[moveSelect] ))
        #per_c = np.float32(beforeImg + oneMv*(getFream-blockFream))
        mat_c = cv2.getPerspectiveTransform(per2, per_c)
        chars_scene = cv2.warpPerspective(chars, mat_c, (imgW, imgH))
        # 背景シーンに挿入
        scene = back.copy()
        for i in range(3):
            scene_i, charsS_i = scene[:, :, i], chars_scene[:, :, i]
            scene_i[charsS_i != 0] = charsS_i[charsS_i != 0]
            scene[:, :, i] = scene_i
        return scene

##########################
########## MAIN ##########
##########################
if __name__ == '__main__':
    #test0()
    #test1()
    #test2()
    test3()
