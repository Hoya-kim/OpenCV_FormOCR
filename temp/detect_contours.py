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
        mid_b = (mid_b / j) * 0.8   # strengthen darkness
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

    cv2.imshow('coutour_first', contour_image)
    cv2.imwrite('../data/result/enhanced_img.png', contour_image)
    cv2.waitKey(0)
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
            img = cv2.rectangle(img, (x, y), (x + width, y + height), (255,255,255, 50), 6)
        if (width > min_width and height > min_height) and ((hierarchy[0, i, 2] != -1 or hierarchy[0, i, 3] == (
                len(hierarchy) - 2 or len(hierarchy) - 1)) or cv2.arcLength(contour, True) > 1000):
            line_image = cv2.rectangle(line_image, (x, y), (x + width, y + height), (255, 255, 255, 50), 2)
            img = cv2.rectangle(img, (x, y), (x + width, y + height), (255,255,255, 50), 6)
        i += 1

    # image that will be erased with white color
    erased_line = cv2.addWeighted(img, 1, line_image, 1, 0)

    cv2.imshow('line image', line_image)
    cv2.imshow('erased image', erased_line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('../data/result/detect_contour.png', img)
    # 지운결과값
    cv2.imwrite('../data/result/erase_lines.png', erased_line)
    # 라인값
    cv2.imwrite('../data/result/line_image.png', line_image)


detect_contours()
