import numpy as np
import cv2
import math

def hough(thr):
    img = cv2.imread('./data/FO-영수증-01.png')
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(imgray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, thr)

    grad0 = math.sin(math.radians(0))
    grad90 = math.sin(math.radians(90))

    for line in lines:
        r, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        #조오오오오올라 잘못됨
        '''
        if(x1 != x2):
            gradient = (y2-y1) / (x2-x1)
            if gradient == grad90 or gradient == grad0:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        else:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        '''

    cv2.imshow('res', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def morph():
    img = cv2.imread('../data/FO-계산서-01.jpg', cv2.IMREAD_GRAYSCALE)

    # 3x3 1로 채워진 매트릭스, Erosion, Dilation을 위한 커널
    kernel = np.ones((3, 3), np.uint8)

    if img is None:
        exit()

    erosion = cv2.erode(img, kernel, iterations=1)  # 원본이미지, 커널, 반복횟수
    dilation = cv2.dilate(img, kernel, iterations=1)

    cv2.imshow('original', img)
    cv2.imshow('erosion', erosion)
    cv2.imshow('dilation', dilation)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def morph2():
    img = cv2.imread('./data/FO-계산서-01.jpg', cv2.IMREAD_GRAYSCALE)

    kernel = np.ones((5, 5), np.uint8)

    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


#hough(200)
#morph()
morph2()
