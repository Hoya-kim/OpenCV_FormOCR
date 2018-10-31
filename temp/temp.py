import numpy as np
import cv2
import math
import random
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim
import pytesseract as ocr


# img : Original image
# line_image : 라인만 있는 이미지
# erased_line : 라인을 지운 이미지

def temp():
    # detect contours
    img = cv2.imread('../data/FO-회원가입-01.png')

    if img is None:
        exit()

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

    # first contour for making line clarify
    for contour in contours:
        # top-left vertex coordinates (x,y) , width, height
        x, y, width, height = cv2.boundingRect(contour)

        # adjacent pixels bgr median, darkness++
        mid_b = 0
        mid_g = 0
        mid_r = 0
        j = 0
        for my in range(y - 2, y + 3):
            for mx in range(x - 2, x + 3):
                if (my and mx) >= 0:
                    mid_b += int(img.item(my, mx, 0))  # b
                    mid_g += int(img.item(my, mx, 1))  # g
                    mid_r += int(img.item(my, mx, 2))  # r
                    j += 1
        mid_b = (mid_b / j) * 0.8
        mid_g = (mid_g / j) * 0.8
        mid_r = (mid_r / j) * 0.8

        # Draw rectangle with median bgr
        # larger than the half of original image size
        if width > origin_width * 0.5 or height > origin_height * 0.5:
            contour_image = cv2.rectangle(img, (x + 1, y + 1), (x + width - 1, y + height - 1),
                                          (mid_b, mid_g, mid_r, 50), 2)
            i += 1
            continue
        if (width > min_width and height > min_height) and ((hierarchy[0, i, 2] != -1 or hierarchy[0, i, 3] == (
                len(hierarchy) - 2 or len(hierarchy) - 1)) or cv2.arcLength(contour, True) > 1000):
            # child가 없는 contour(최내곽)이 아니거나 parent가 최외곽 or 최외곽-1인 contour
            # 또는 arc length > 1000 이상
            contour_image = cv2.rectangle(img, (x - 1, y - 1), (x + width + 1, y + height + 1),
                                          (mid_b, mid_g, mid_r, 50), 2)
        i += 1

    # cv2.imwrite('./data/result/boxing_clarify.png', contour_image)
    # cv2.imshow('coutour_first', contour_image)
    # cv2.waitKey(0)
    ########################################################################################################################
    # ocr box by box
    for_ocr = np.copy(contour_image)
    for_ocr2 = np.copy(contour_image)
    imgray = cv2.cvtColor(for_ocr, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(imgray, 230, 255, 0)
    _, contours, hierarchy = cv2.findContours(thr, cv2.RETR_CCOMP,
                                              cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_CCOMP --> hirarchy level을 2단계로만 구성
    # bounding cell #글자 내에 생기는 contour는 작음
    min_width = 15
    min_height = 15
    i = 0

    for contour in contours:
        # top-left vertex coordinates (x,y) , width, height
        x, y, width, height = cv2.boundingRect(contour)
        # Draw screenshot that are larger than the standard size
        if (width > min_width or height > min_height) and ((hierarchy[0, i, 2] != -1 or hierarchy[0, i, 3] == (len(hierarchy) - 2 or len(hierarchy) - 1)) or cv2.arcLength(contour, True) > 1000):
            # child가 없는 contour(최내곽)이 아니거나 parent가 최외곽 or 최외곽-1인 contour
            temp_img = (for_ocr[y:y + height, x:x + width])
            if (width > 5 or height > 5):
                text = ocr.image_to_string(temp_img, lang='kor')
                if (len(text) == 0):
                    pass
                else:
                    print(text)
                    cv2.imshow(str(i), temp_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            # 중복으로 OCR 되는 것을 막기 위해 하얀색으로 채워줌
            cv2.rectangle(for_ocr, (x, y), (x + width, y + height), (255, 255, 255), cv2.FILLED)
        i += 1
    '''
    for tmp_img in tmp_images:
        tmp_imgray = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
        ret, thr = cv2.threshold(tmp_imgray, 230, 255, 0)
        _, tmp_contours, _ = cv2.findContours(thr, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_CCOMP --> hirarchy level을 2단계로만 구성
        for tmp_contour in tmp_contours:
            x, y, width, height = cv2.boundingRect(tmp_contour)
            # Draw screenshot that are larger than the standard size
            if width > min_width or height > min_height :
                tmp_img22 = (tmp_img[y:y + height, x:x + width])
                text = ocr.image_to_string(tmp_img22, lang='kor')
                if (len(text) == 0):
                    pass
                else:
                    print(text)
                    cv2.imshow(str(i), tmp_img22)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                # 중복으로 OCR 되는 것을 막기 위해 하얀색으로 채워줌
                cv2.rectangle(tmp_img, (x, y), (x + width, y + height), (255, 255, 255), cv2.FILLED)
    '''
    ########################################################################################################################
    # repeat contour detection
    imgray = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(imgray, 230, 255, 0)

    _, contours, hierarchy = cv2.findContours(thr, cv2.RETR_CCOMP,
                                              cv2.CHAIN_APPROX_SIMPLE)

    # image that only lines will be drawn
    line_image = contour_image * 0
    i = 0

    # Draw rectangle with white color
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        if width > origin_width * 0.5 or height > origin_height * 0.5:
            line_image = cv2.rectangle(line_image, (x, y), (x + width, y + height), (255, 255, 255, 50), 2)
        if (width > min_width and height > min_height) and ((hierarchy[0, i, 2] != -1 or hierarchy[0, i, 3] == (
                len(hierarchy) - 2 or len(hierarchy) - 1)) or cv2.arcLength(contour, True) > 1000):
            line_image = cv2.rectangle(line_image, (x, y), (x + width, y + height), (255, 255, 255, 50), 2)
        i += 1

    # image that will be erased with white color
    erased_line = cv2.addWeighted(img, 1, line_image, 1, 0)

    ########################################################################################################################

    # 1) line image closing
    # 2) detect_contour with
    # 3) calculte num of cells

    # line_image = cv2.imread('./data/result/line_image.png')

    kernel = np.ones((3, 3), np.uint8)

    line_image = cv2.morphologyEx(line_image, cv2.MORPH_CLOSE, kernel, iterations=5)

    imgray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(imgray, 230, 255, 0)

    _, contours, _ = cv2.findContours(thr, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)

    # for calculate that num of cells needed
    needed_x = []
    needed_y = []
    minimum_w = 1000
    minimum_h = 1000

    for contour in contours:
        # top-left vertex coordinates (x,y) , width, height
        x, y, width, height = cv2.boundingRect(contour)
        if minimum_w > width and width > 15:
            minimum_w = width
        if minimum_h > height and height > 15:
            minimum_h = height
        # Draw screenshot that are larger than the standard size
        if width > min_width or height > min_height:
            line_image = cv2.rectangle(line_image, (x, y), (x + width, y + height), (255, 0, 0, 50), 2)
            needed_x.append(x)
            needed_x.append(x + width)
            needed_y.append(y)
            needed_y.append(y + height)

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
            if (temp_int == needed_x[a - 1]) and a != 1:
                final_x.add(temp_int)
            else:
                final_x.add(int(temp_int / num_temp_int))
            num_temp_int = 1
            temp_int = needed_x[a]
    final_x.add(int(temp_int / num_temp_int))

    temp_int = needed_y[0]
    num_temp_int = 1
    for a in range(1, len(needed_y)):
        if needed_y[a] - needed_y[a - 1] < minimum_h:
            temp_int += needed_y[a]
            num_temp_int += 1
        else:
            if (temp_int == needed_y[a - 1]) and a != 1:
                final_y.add(temp_int)
            else:
                final_y.add(int(temp_int / num_temp_int))
            num_temp_int = 1
            temp_int = needed_y[a]
    final_y.add(int(temp_int / num_temp_int))

    print(needed_y)  # final_x - 1개의 셀이 필요함
    print(needed_y)  # final_y - 1개의 셀이 필요함

    print('x축 셀의 갯수', len(final_x) - 1)
    print(sorted(list(final_x)))
    print('y축 셀의 갯수', len(final_y) - 1)
    print(sorted(list(final_y)))

    for x in final_x:
        cv2.line(line_image, (x, 0), (x, origin_height), (0, 0, 255), 2)
    for y in final_y:
        cv2.line(line_image, (0, y), (origin_width, y), (0, 0, 255), 2)

    # cv2.line(closing, (1020, 0), (1020, origin_height), (0, 255, 0), 2)

    cv2.imshow('draw_needed_cell', line_image)
    #cv2.imwrite('./data/result/cal_cell_needed.png', line_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


temp()
