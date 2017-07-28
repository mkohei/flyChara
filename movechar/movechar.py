# encoding=utf-8

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


# -------------------------------
# ---------- CONSTANTS ----------
# -------------------------------
# TITLE
TITLE = 'movechar'

# threshold
th = 180

# PAPER SIZE
N = 3
H, W = 210*N, 297*N

# COLOR
RED, GREEN, BLUE = (0, 0, 255), (0, 255, 0), (255, 0, 0)
# CORNER
INIT_CORNER = [[666,458], [1347,444], [580,771], [1462,758]]
R = 50



# -------------------------------
# ---------- VARIABLES ----------
# -------------------------------
corner = []
img = None
imgH, imgW = None, None

# move
cornerCnt0 = []
goalX, goalY = [0, 0, 0, 0], [0, 0, 0, 0]
stepX, stepY = [0, 0, 0, 0], [0, 0, 0, 0]
fps=30

corner_ = []
# -----------------------------
# ---------- METHODS ----------
# -----------------------------
def move1(corner, cnt):
    global cornerCnt0
    global chars, back, per2, imgW, imgH
    if cnt == 0:
        cornerCnt0 = []
        for i, c in enumerate(corner):
            cornerCnt0.append([c[0], c[1]])
            corner_.append([c[0], c[1]])
        goalX[0], goalY[0] = corner[0][0], corner[0][1]+200
        goalX[1], goalY[1] = corner[1][0], corner[0][1]+200
        goalX[2], goalY[2] = corner[0][0], corner[0][1]-400+200
        goalX[3], goalY[3] = corner[1][0], corner[0][1]-400+200

        for i in range(4):
            stepX[i] = (goalX[i] - cornerCnt0[i][0]) / fps
            stepY[i] = (goalY[i] - cornerCnt0[i][1]) / fps
    if cnt < fps:
        for i in range(4):
            corner_[i][0] += stepX[i]
            corner_[i][1] += stepY[i]
    else:
        if cnt == fps:
            cornerCnt0 = []
        for i, c in enumerate(corner_):
            cornerCnt0.append([c[0], c[1]])
        for i in range(4):
            corner_[i][0] -= 10
            corner_[i][1] = cornerCnt0[i][1]-100*np.abs(np.sin(cnt/10*np.pi))
    mat_c = cv2.getPerspectiveTransform(per2, np.float32(corner_))
    chars_scene = cv2.warpPerspective(chars, mat_c, (imgW, imgH))
    scene = back.copy()
    for i in range(3):
        scene_i, charsS_i = scene[:, :, i], chars_scene[:, :, i]
        scene_i[charsS_i != 0] = charsS_i[charsS_i != 0]
        scene[:, :, i] = scene_i
    return scene



# ----------------------------------
# ---------- MAIN METHODS ----------
# ----------------------------------
def calib(img):
    global corner, INIT_CORNER
    corner = []
    # コーナー検出
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    corner_points = np.where(dst > 0.01*dst.max())
    ypoints, xpoints = corner_points

    # 矩形検
    for p in INIT_CORNER:
        d = (p[0]-xpoints)**2 + (p[1]-ypoints)**2
        i = np.argmin(d)
        corner.append([xpoints[i], ypoints[i]])
        # 検出コーナー
        cv2.circle(img, (xpoints[i], ypoints[i]), 10, BLUE, -1)
        # 定義コーナー
        cv2.circle(img, (p[0], p[1]), 50, RED, 5)
    # 検出矩形
    if len(corner) > 3:
        cv2.line(img, (corner[0][0],corner[0][1]), (corner[1][0],corner[1][1]), BLUE, 3)
        cv2.line(img, (corner[0][0],corner[0][1]), (corner[2][0],corner[2][1]), BLUE, 3)
        cv2.line(img, (corner[1][0],corner[1][1]), (corner[3][0],corner[3][1]), BLUE, 3)
        cv2.line(img, (corner[2][0],corner[2][1]), (corner[3][0],corner[3][1]), BLUE, 3)

    return img


inpkt = [-1,-1,-1,-1]
inpktPoint = [[],[],[],[]]
def animeFream(corners, getFream):
    global inpkt
    global inpktPoint
    global per2, chars, back, imgW, imgH
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



# --------------------------
# ---------- MAIN ----------
# --------------------------
# カメラキャプチャ
cap = cv2.VideoCapture(1)

### キャリブレーション
while True:
    # load
    ret, frame = cap.read()
    img = frame.copy()
    imgH, imgW = img.shape[:2]

    frame = calib(frame)
    cv2.imshow(TITLE, frame)
    key = cv2.waitKey(0)
    if key == 13: break # Enter


# 少し内側に

corner[0][0] += 10
corner[0][0] += 10
corner[1][0] -= 10
corner[1][1] += 10
corner[2][0] += 20
corner[2][1] -= 20
corner[3][0] -= 20
corner[3][1] -= 20


### 紙領域の抽出
corner0 = []
for i in range(4):
    corner0.append([corner[i][0], corner[i][1]])
print(corner, corner0)
per1 = np.float32(corner)
per2 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
mat12 = cv2.getPerspectiveTransform(per1, per2) # 1->2
paper = cv2.warpPerspective(img, mat12, (W, H))

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


# シーンに文字を戻す
chars_scene = cv2.warpPerspective(chars, mat21, (imgW, imgH))
chars_scene_ = chars_scene.copy()


### アニメーションスタート
cnt = 0
func_cnt = 0
FUNC = [animeFream, move1]
while True:
    scene = FUNC[func_cnt](corner, cnt)
    cnt += 1
    func_cnt = (func_cnt + (cnt//160)) % len(FUNC)
    cnt = cnt % 160
    cv2.imshow(TITLE, scene)
    cv2.waitKey(1)

### 
#cv2.imshow(TITLE, chars)
cv2.imshow(TITLE, scene)
cv2.waitKey(0)
