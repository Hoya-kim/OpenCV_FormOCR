import numpy as np
import cv2


def detect_contours():
    # detect contours
    img = cv2.imread('../data/FO-회원가입-01.png')
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.RETR_CCOMP --> hirarchy level을 2단계로만 구성
    ret, thr = cv2.threshold(imgray, 230, 255, 0)
    _, contours, hierarchy = cv2.findContours(thr, cv2.RETR_CCOMP,
                                              cv2.CHAIN_APPROX_SIMPLE)
    # original image's width & height
    origin_width = img.shape[1]
    origin_height = img.shape[0]

    # 글자 내에 생기는 contour는 작음
    # 실제 런칭시에는 사용자에게 기본폰트 사이즈가 몇인지 입력받는다
    # font_size + 1
    min_width = 15
    min_height = 15
    i = 0

    # creating a blank to draw lines on
    contour_image = np.copy(img) * 0

    # first contour detection for making line clarify
    for contour in contours:
        # top-left vertex coordinates (x,y) , width, height
        x, y, width, height = cv2.boundingRect(contour)

        # Draw rectangle with median bgr
        # larger than the half of original image size
        if width > min_width and height > min_height:
            # child가 없는 contour(최내곽)이 아니거나 parent가 최외곽 or 최외곽-1인 contour
            # 또는 arc length > 1000 이상
            contour_image = cv2.rectangle(img, (x, y), (x + width, y + height),
                                          (0,255,0, 50), 2)
        i += 1

    cv2.imshow('coutour_first', contour_image)
    cv2.imwrite('../data/result/그냥Contour.png', contour_image)
    cv2.waitKey(0)


detect_contours()