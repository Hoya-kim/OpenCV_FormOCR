import cv2
import numpy as np
import math
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt

def detect_line():
    # ============================================================
    # First, get the gray image and process GaussianBlur.
    img = cv2.imread('../data/FO-회원가입-01.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #kernel_size = 3
    #blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # ============================================================
    # Second, process edge detection use Canny.
    low_threshold = 70
    high_threshold = 200
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # ============================================================
    # Then, use HoughLinesP to get the lines.
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 150  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 200  # minimum number of pixels making up a line
    max_line_gap = 150  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    #grad0 = math.sin(math.radians(0))
    #grad90 = math.sin(math.radians(90))

    #i=0
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #실제 x,y값은 보이는 것 보다 많다
            #print(i, 'x : ',x1,', ',x2)
            #print('y : ', y1, ', ', y2, '\n')
            #i+=1

        # 기울기 0 or inf 인 데이터들만 라인 따기
        # if (x1 != x2):
        #    gradient = abs((y2-y1) / (x2-x1))
        #    if gradient == grad0 or gradient == grad90:
        #        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
        # else:   #세로축
        #    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # ============================================================
    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 1, line_image, 1, 0)
    cv2.imshow("lines_edges", lines_edges)
    cv2.waitKey(0)

    # 지운결과값
    cv2.imwrite('../data/result/hough.png', lines_edges)
    # 라인값
    #cv2.imwrite('./data/result/line_image.png', line_image)

def morph(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # 3x3 1로 채워진 매트릭스, Erosion, Dilation을 위한 커널
    kernel = np.ones((3, 3), np.uint8)

    if img is None:
        exit()

    erosion = cv2.erode(img, kernel, iterations=1)  # 원본이미지, 커널, 반복횟수
    dilation = cv2.dilate(img, kernel, iterations=1)

    cv2.imshow('original', img)
    cv2.imshow('erosion', erosion)
    cv2.imshow('dilation', dilation)
    cv2.imshow('orginal2', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def morph2(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


detect_line()
#morph2('./data/result/erase_lines.png')
#morph2('./data/result/line_image.png')