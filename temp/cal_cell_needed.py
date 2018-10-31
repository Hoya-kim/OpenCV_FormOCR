import numpy as np
import cv2
import xlsxwriter
from string import ascii_uppercase
import os
import yaml


def cal_cell_needed():
    # #######################################################################################################################

    # 1) line image closing
    # 2) detect_contour with closing image
    # 3) calculte num of cells

    line_image = cv2.imread('../data/result/line_image.png')

    # original image's width & height
    origin_width = line_image.shape[1]
    origin_height = line_image.shape[0]

    kernel = np.ones((3, 3), np.uint8)

    line_image = cv2.morphologyEx(line_image, cv2.MORPH_CLOSE, kernel, iterations=2)

    imgray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(imgray, 230, 255, 0)

    _, contours, _ = cv2.findContours(thr, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)

    # for calculate that num of cells needed
    min_width = 15
    min_height = 15
    needed_x = []
    needed_y = []
    minimum_w = 1000
    minimum_h = 1000

    for contour in contours:
        # top-left vertex coordinates (x,y) , width, height
        x, y, width, height = cv2.boundingRect(contour)
        if minimum_w > width and width > min_width:
            minimum_w = width
        if minimum_h > height and height > min_height:
            minimum_h = height
        # Draw screenshot that are larger than the standard size
        if width > min_width or height > min_height:
            line_image = cv2.rectangle(line_image, (x, y), (x + width, y + height), (255, 0, 0, 50), 2)
            needed_x.append(x)
            needed_x.append(x + width)
            needed_y.append(y)
            needed_y.append(y + height)

    tmp_img = np.copy(line_image)

    for x in needed_x:
        cv2.line(tmp_img, (x, 0), (x, origin_height), (0, 0, 255), 2)
    for y in needed_y:
        cv2.line(tmp_img, (0, y), (origin_width, y), (0, 0, 255), 2)

    cv2.imwrite('../data/result/임시.png', tmp_img)
    print('minimun_w: ', minimum_w)
    print('minimun_h: ', minimum_h)
    # ========================================================
    # num of needed cells
    needed_x = sorted(list(set(needed_x)))  # list(set(my_list)) --> 중복제거
    needed_y = sorted(list(set(needed_y)))

    final_x = set()
    final_y = set()
    temp_int = needed_x[0]
    num_temp_int = 1
    for a in range(1, len(needed_x)):
        if needed_x[a] - needed_x[a - 1] < minimum_w:
            temp_int += needed_x[a]
            num_temp_int += 1
        else:
            final_x.add(int(temp_int / num_temp_int))
            num_temp_int = 1
            temp_int = needed_x[a]
    final_x.add(int(temp_int / num_temp_int))
    if min(final_x) > minimum_w:
        final_x.add(0)
    if origin_width - max(final_x) > minimum_w:
        final_x.add(origin_width)

    temp_int = needed_y[0]
    num_temp_int = 1
    for a in range(1, len(needed_y)):
        if needed_y[a] - needed_y[a - 1] < minimum_h:
            temp_int += needed_y[a]
            num_temp_int += 1
        else:
            final_y.add(int(temp_int / num_temp_int))
            num_temp_int = 1
            temp_int = needed_y[a]
    final_y.add(int(temp_int / num_temp_int))
    if min(final_y) > minimum_h:
        final_x.add(0)
    if origin_height - max(final_y) > minimum_h:
        final_x.add(origin_height)

    print(needed_x)  # final_x - 1개의 셀이 필요함
    print(needed_y)  # final_y - 1개의 셀이 필요함
    final_x = sorted(list(final_x))
    final_y = sorted(list(final_y))
    print('x축 셀의 갯수', len(final_x) - 1)
    print(final_x)
    print('y축 셀의 갯수', len(final_y) - 1)
    print(final_y)

    for x in final_x:
        cv2.line(line_image, (x, 0), (x, origin_height), (0, 0, 255), 2)
    for y in final_y:
        cv2.line(line_image, (0, y), (origin_width, y), (0, 0, 255), 2)

    cv2.imshow('cal_cell_needed', line_image)
    cv2.imwrite('../data/result/temp.png', line_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.line(closing, (1020, 0), (1020, origin_height), (0, 255, 0), 2)
    # 이 나눈셀들을 바탕으로 셀의 정보를 할당해야함

    # cell_dict
    # 셀의 이름
    # upper_left의 (x,y)의 좌표
    # lower_right의 좌표를 구하기 위한 (width, height)
    # OCR한 결과 값을 넣을 string(text)이 필요할듯?

    # cell_dict = {(x,y) for x in final_x for y in final_y}
    cell_dict = dict()
    for i in range(0, len(final_x)):
        for j in range(0, len(final_y)):
            tmp = ascii_uppercase[i] + "%d" % j
            cell_dict[tmp] = (final_x[i], final_y[j])

    print(cell_dict)



cal_cell_needed()
