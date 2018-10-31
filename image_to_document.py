#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Jungho Kim (Hoya_kim)"
import os, sys
import cv2
import yaml
import xlsxwriter
import numpy as np
from string import ascii_uppercase
import pytesseract as ocr
import time

################################################################################
class Cell(object):
    """ Form에서 이루어지는 각각의 cell들의 대한 정보를 저장합니다. """

    def __init__(self):
        # cells matrix
        self.x = None
        self.y = None
        self.width = None
        self.height = None

        self.central_x = None
        self.central_y = None

        # text info
        self.text = None
        self.text_height = None
        self.text_align = 'center'
        self.text_valign = 'vcenter'

        self.cell_name = None
        self.merged_info = None
        self.bg_color = '#ffffff'

        self.boundary = {
            'left': False,
            'right': False,
            'upper': False,
            'lower': False
        }

    # ==========================================================================
    def _set_value(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.central_x = x + int(width / 2)
        self.central_y = y + int(height / 2)

    # ==========================================================================
    def _get_value(self):
        return self.x, self.y, self.width, self.height, self.central_x, self.central_y

    # ==========================================================================
    def _set_text(self, text, text_height, text_align, text_valign):
        self.text = text
        self.text_height = text_height
        self.text_align = text_align
        self.text_valign = text_valign

    # ==========================================================================
    def _merge_horizontal(self, right_cell):
        """ 가로로 정렬된 Cell들을 merge합니다.
        :param right_cell: 현재 Cell을 기준으로 우측에 위치한 Cell
        """
        self.width += right_cell.width
        self.central_x = self.x + int(self.width / 2)
        self.boundary['right'] = right_cell.boundary['right']
        self.merged_info = right_cell.merged_info

    # ==========================================================================
    def _merge_vertical(self, lower_cell):
        """ 가로로 정렬된 Cell들을 merge합니다.
        :param lower_cell: 현재 Cell을 기준으로 하단에 위치한 Cell
        """
        self.height += lower_cell.height
        self.central_y = self.y + int(self.height / 2)
        self.boundary['lower'] = lower_cell.boundary['lower']
        self.merged_info = lower_cell.merged_info

    # ==========================================================================
    def __repr__(self):
        """ Console로 Cell이 지닌 속성들을 print합니다. """
        s = str()
        s += 'x %d y %d\t|\t' % (self.x, self.y)
        s += 'w %d h %d\t|\t' % (self.width, self.height)

        s += self.cell_name + '\t|\t'

        if self.text is not None:
            s += self.text
        else:
            s += 'None'

        s += '\t\theight: ' + str(self.text_height)
        s += '\talign/valign: ' + self.text_align + '/' + self.text_valign

        s += '\t\t'
        s += str(self.boundary)

        return s


################################################################################
class Preprocessing(object):
    def __init__(self, img_file, conf_file=None, verbose='v'):

        if img_file:
            if not os.path.exists(img_file):
                raise IOError('Cannot find image file "%s"' % img_file)
        self.img_file = img_file
        self.img = cv2.imread(img_file)  # 작업용 이미지

        ####### image_warping 전처리 ############
        #self.img = self.image_warping()
        #self.img = self.resize_image(self.img)
        #######################################

        #self.Origin_image = cv2.imread(img_file)
        self.Origin_image = self.img.copy()
        self.line_image = self.img * 0  # image that only lines will be drawn
        self.erased_line = None  # image that will be erased  with white color
        self.closing_line = None

        # original image's width & height
        self.origin_height, self.origin_width = self.Origin_image.shape[:2]

        if not conf_file:
            # 디폴트는 현재 패키지 위치의 config.yaml 파일을 읽음
            conf_file = '%s/config.yaml' \
                        % os.path.abspath(os.path.dirname(__file__))
            if not os.path.exists(conf_file):
                raise IOError('Cannot find config file "%s"' % conf_file)
        self.config_str = None
        self.config = self._read_config(conf_file)

        # 꼭 필요한 row, col에 대한 리스트
        self.final_x = None
        self.final_y = None

        # 추출된 셀들 사이 가장 작은 width, height
        self.find_min_width = None
        self.find_min_height = None

        # cell의 정보를 가지고 있음
        self.cells = None
        self.before_merged = None

        self.verbose = verbose

    # ==========================================================================
    def image_warping(self):
        orig = self.img.copy()
        r = 800.0 / self.img.shape[0]
        dim = (int(self.img.shape[1] * r), 800)
        self.img = cv2.resize(self.img, dim, interpolation=cv2.INTER_AREA)

        imgray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        imgray = cv2.GaussianBlur(imgray, (3, 3), 0)
        edged = cv2.Canny(imgray, 75, 200)

        _, contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

        image = np.copy(self.img)

        for contour in contours:
            perl = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perl, True)

            if len(approx) == 4:
                screenCnt = approx
                break

        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
        cv2.imshow('find_Contour', image)
        cv2.waitKey(0)

        rect = self.order_points(screenCnt.reshape(4, 2) / r)

        (topLeft, topRight, bottomRight, bottomLeft) = rect

        w1 = abs(bottomRight[0] - bottomLeft[0])
        w2 = abs(topRight[0] - topLeft[0])
        h1 = abs(topRight[1] - bottomRight[1])
        h2 = abs(topLeft[1] - bottomLeft[1])

        maxWidth = max([w1, w2])
        maxHeight = max([h1, h2])

        dst = np.float32([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]])
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

        cv2.imshow('warping', warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return warped

    # ==========================================================================
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)

        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    # ==========================================================================
    def resize_image(self, img):
        """ 촬영된 이미지는 사이즈가 큼 """
        if img.shape[0] > 2000 or img.shape[1] > 2000:
            r = 1500.0 / self.img.shape[0]
            dim = (int(self.img.shape[1] * r), 1500)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow('resize', img)
            cv2.waitKey(0)
        return img

    # ==========================================================================
    def _read_config(self, config_file):
        """ .yaml file 을 읽어서 configuration 값의 객체를 갖습니다.

        :param config_file:
        :return: 읽은 configuration 을 담고있는 dictionary 형태로 반환
        """
        # read contents from .yam config file
        with open(config_file, 'r', encoding='UTF-8') as ifp:
            self.config_str = ifp.read()
        with open(config_file, 'r', encoding='UTF-8') as ifp:
            return yaml.load(ifp)

    # ==========================================================================
    def _show_img(self, title, target_img):
        temp_img = np.copy(target_img)
        if self.verbose.startswith('vv'):
            if self.origin_width > 1000 or self.origin_height > 1000:
                temp_img = cv2.resize(temp_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            cv2.imshow(title, temp_img)
            cv2.waitKey(0)
            cv2.imwrite('./data/result/' + title + '.png', temp_img)

    # ==========================================================================
    def _get_threshold(self, imgray, mode):
        """ 이미지에 Threshold 를 적용해서 이진(Binary) 이미지 객체를 반환합니다.
        이미지의 글자와 line이 적절히 나누어 지도록 mode에 따라 parameter값이 달라집니다.
        :param mode: mode에 따라서 low, high_threshold의 값이 달라집니다.
        """
        low_threshold = self.config[mode]['low_threshold']
        high_threshold = self.config[mode]['high_threshold']
        thr_type = self.config[mode]['thr_type']

        ret, thr = cv2.threshold(imgray, low_threshold, high_threshold, thr_type)
        # th2 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
        # th3 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
        return thr

    # ==========================================================================
    def boxing_ambiguous(self):
        """ 이미지에서 line이 위아래만 적용되어 있거나, 경계선이 그려져 있지 않고 색상으로 경계된 Cell등의
        모호한 경계에 대해 강제로 경계를 그려줍니다.
        """
        imgray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        mode = 'boxing'
        thr = self._get_threshold(imgray, mode)

        min_width = self.config['contour']['min_width']
        min_height = self.config['contour']['min_height']
        retrieve_mode = self.config['contour']['retrieve_mode']
        approx_method = self.config['contour']['approx_method']
        _, contours, hierarchy = cv2.findContours(thr, retrieve_mode, approx_method)

        i = 0
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
                    if (my and mx) >= 0 and (my < self.img.shape[0]) and (mx < self.img.shape[1]):
                        mid_b += int(self.img.item(my, mx, 0))  # b
                        mid_g += int(self.img.item(my, mx, 1))  # g
                        mid_r += int(self.img.item(my, mx, 2))  # r
                        j += 1
            mid_b = (mid_b / j) * 0.7  # strengthen darkness
            mid_g = (mid_g / j) * 0.7
            mid_r = (mid_r / j) * 0.7

            # Draw rectangle with median bgr
            # larger than the half of original image size
            if width > self.origin_width * 0.5 or height > self.origin_height * 0.5:
                self.img = cv2.rectangle(self.img, (x + 1, y + 1), (x + width - 1, y + height - 1),
                                         (mid_b, mid_g, mid_r, 50), 2)
                i += 1
                continue
            if (width > min_width and height > min_height) and ((hierarchy[0, i, 2] != -1 or hierarchy[0, i, 3] == (
                    len(hierarchy) - 2 or len(hierarchy) - 1)) or cv2.arcLength(contour, True) > 1000):
                # child가 없는 contour(최내곽)이 아니거나 parent가 최외곽 or 최외곽-1인 contour
                # 또는 arc length > 1000 이상
                self.img = cv2.rectangle(self.img, (x - 1, y - 1), (x + width + 1, y + height + 1),
                                         (mid_b, mid_g, mid_r, 50), 2)
            i += 1
        self._show_img('_strengthen_img', self.img)

    # ==========================================================================
    def detect_contours(self):
        """ 경계선이 강화된 이미지에 다시 한 번 Contour(윤곽)을 찾아 표에 대한 경계를 찾아냅니다.
        찾아낸 경계를 까만(b,g,r = 0)이미지에 흰색(b,g,r = 255)의 사각형을 그려 line만 있는 line_image를 만듭니다.
        """
        # repeat contour detection
        imgray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        mode = 'detect'
        thr = self._get_threshold(imgray, mode)

        min_width = self.config['contour']['min_width']
        min_height = self.config['contour']['min_height']
        retrieve_mode = self.config['contour']['retrieve_mode']
        approx_method = self.config['contour']['approx_method']
        _, contours, hierarchy = cv2.findContours(thr, retrieve_mode, approx_method)

        i = 0
        # Draw rectangle with white color
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)

            if width > self.origin_width * 0.5 or height > self.origin_height * 0.5:
                self.line_image = cv2.rectangle(self.line_image, (x, y), (x + width, y + height), (255, 255, 255, 50),
                                                2)
            if (width > min_width and height > min_height) and ((hierarchy[0, i, 2] != -1 or hierarchy[0, i, 3] == (
                    len(hierarchy) - 2 or len(hierarchy) - 1)) or cv2.arcLength(contour, True) > 1000):
                self.line_image = cv2.rectangle(self.line_image, (x, y), (x + width, y + height), (255, 255, 255, 50),
                                                2)
            i += 1

        self._show_img('_line_img', self.line_image)

    # ==========================================================================
    def erase_line(self):
        """ 흰색으로 그려진 Line_image를 Original_imgae에 덮어 씌워 경계를 지워줍니다."""
        # image that will be erased with white color
        # self.closing_line = cv2.cvtColor(self.closing_line, cv2.COLOR_GRAY2BGR)
        self.erased_line = cv2.addWeighted(self.Origin_image, 1, self.closing_line, 1, 0)
        self._show_img('_erased_img', self.erased_line)

    # ==========================================================================
    '''
    def detect_line(self):
        img = self.line_image

        low_threshold = self.config['canny']['low_threshold']
        high_threshold = self.config['canny']['high_threshold']
        edges = cv2.Canny(img, low_threshold, high_threshold)

        rho = self.config['houghline']['rho']  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = self.config['houghline']['threshold']  # minimum number of votes (intersections in Hough grid Cell)
        min_line_length = min(self.origin_height, self.origin_width) * 0.2  # minimum number of pixels making up a line
        max_line_gap = min(self.origin_height,
                           self.origin_width) * 0.08  # maximum gap in pixels between connectable line segments

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2 or y1 == y2:
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        self._show_img('detect_line', self.line_image)
    '''

    # ==========================================================================
    def morph_closing(self):
        """ Line_image에서 Line과 Line사이에 존재하는 공간을 Morph Close기법을 이용하여 매꿔줍니다.
        Line과 Line사이에 존재하는 공간은 실제 라인이 존재하는 공간이기 때문에 필요한 Cell들을 계산할 때 필요하지 않습니다.
        """
        # for line_image
        kernel_row = self.config['closing']['kernel_size_row']
        kernel_col = self.config['closing']['kernel_size_col']
        kernel = np.ones((kernel_row, kernel_col), np.uint8)

        closing_iter = self.config['closing']['iteration']

        self.closing_line = cv2.morphologyEx(self.line_image, cv2.MORPH_CLOSE, kernel, iterations=closing_iter)

        self._show_img('_closing_line_img', self.closing_line)

    # ==========================================================================
    def cal_cell_needed(self):
        """ Closing 된 line_image에서 다시 한 번 cv2.findContour()를 적용하여 필요한 x,y axis 들을 계산합니다.
        """
        imgray = cv2.cvtColor(self.closing_line, cv2.COLOR_BGR2GRAY)

        mode = 'default_threshold'
        thr = self._get_threshold(imgray, mode)

        min_width = self.config['contour']['min_width']
        min_height = self.config['contour']['min_height']
        retrieve_mode = self.config['contour']['retrieve_mode']
        approx_method = self.config['contour']['approx_method']
        _, contours, hierarchy = cv2.findContours(thr, retrieve_mode, approx_method)

        needed_x = []
        needed_y = []
        find_min_width = self.config['num_of_needed_cell']['find_min_width']
        find_min_height = self.config['num_of_needed_cell']['find_min_height']

        for contour in contours:
            # top-left vertex coordinates (x,y) , width, height
            x, y, width, height = cv2.boundingRect(contour)
            if find_min_width > width and width > min_width:
                find_min_width = width
            if find_min_height > height and height > min_height:
                find_min_height = height
            # Draw screenshot that are larger than the standard size
            if width > min_width or height > min_height:
                # self.line_image = cv2.rectangle(self.line_image, (x, y), (x + width, y + height), (255, 0, 0, 50), 2)
                needed_x.append(x)
                needed_x.append(x + width)
                needed_y.append(y)
                needed_y.append(y + height)

        # num of needed cells
        needed_x = sorted(list(set(needed_x)))  # list(set(my_list)) --> 중복제거
        needed_y = sorted(list(set(needed_y)))

        self.find_min_width = find_min_width
        self.find_min_height = find_min_height

        # contour와 contour사이의 선은 cell로 취급할 필요가 없음
        # 가장 작은 contour rectangle의 width와 height를 기준으로 유사한 값을 압축
        self.final_x = self.approx_axis(needed_x, int(find_min_width * 0.5))
        self.final_y = self.approx_axis(needed_y, int(find_min_height * 0.5))

        self.draw_axis()

    # ==========================================================================
    def draw_axis(self):
        if self.verbose.startswith('vv'):
            tmp_img1 = np.copy(self.line_image)
            tmp_img2 = np.copy(self.erased_line)

            for x in self.final_x:
                cv2.line(tmp_img1, (x, 0), (x, self.origin_height), (0, 0, 255), 2)
            for y in self.final_y:
                cv2.line(tmp_img1, (0, y), (self.origin_width, y), (0, 0, 255), 2)
            self._show_img('_draw_axis', tmp_img1)

            for x in self.final_x:
                cv2.line(tmp_img2, (x, 0), (x, self.origin_height), (0, 0, 255), 2)
            for y in self.final_y:
                cv2.line(tmp_img2, (0, y), (self.origin_width, y), (0, 0, 255), 2)
            self._show_img('_draw_axis', tmp_img2)

    # ==========================================================================
    def approx_axis(self, needed_axis, find_min_axis):
        """ 효율적으로 필요한 axis의 개수를 구하기 위해 list의 값들을 압축합니다.
        :param needed_axis: cal_cell_needed 메소드에서 구한 필요한 x 또는 y axis의 list
        :param find_min_axis: x, y axis list를 압축하기 위한 임계 값
        :return: 압축된 x,y axis list
        """
        final_axis = set()
        temp_int = needed_axis[0]
        num_temp_int = 1
        for a in range(1, len(needed_axis)):
            if needed_axis[a] - needed_axis[a - 1] < find_min_axis:
                temp_int += needed_axis[a]
                num_temp_int += 1
            else:
                final_axis.add(int(temp_int / num_temp_int))
                num_temp_int = 1
                temp_int = needed_axis[a]
        final_axis.add(int(temp_int / num_temp_int))
        if min(final_axis) > find_min_axis:
            final_axis.add(0)
        if self.origin_width - max(final_axis) > find_min_axis:
            final_axis.add(self.origin_width)

        final_axis = sorted(list(final_axis))  # len(final_axis) - 1 개의 셀이 필요함
        return final_axis

    # ==========================================================================
    def save_cell_value(self):
        """ 현재까지 추출해낸 각각의 Cell들의 x, y, width, height 의 값들과
        Excel(엑셀)로 게워내기 좀 더 편리하도록 각 Cell에 알파벳과 숫자로 이루어진 이름을 입력합니다.
        """
        self.cells = [[Cell() for rows in range(len(self.final_x) - 1)] for cols in
                      range(len(self.final_y) - 1)]
        self.before_merged = [[Cell() for rows in range(len(self.final_x) - 1)] for cols in
                              range(len(self.final_y) - 1)]

        for cols in range(len(self.final_y) - 1):
            for rows in range(len(self.final_x) - 1):
                x = self.final_x[rows]
                y = self.final_y[cols]
                width = self.final_x[rows + 1] - self.final_x[rows]
                height = self.final_y[cols + 1] - self.final_y[cols]

                self.cells[cols][rows]._set_value(x, y, width, height)
                self.before_merged[cols][rows]._set_value(x, y, width, height)

                self.cells[cols][rows].cell_name = ascii_uppercase[rows] + "%d" % (cols + 1)
                # 본인의 cell_name과 merged_info가 같으면 머지 되지 않은 것
                self.cells[cols][rows].merged_info = ascii_uppercase[rows] + "%d" % (cols + 1)
                self.before_merged[cols][rows].cell_name = ascii_uppercase[rows] + "%d" % (cols + 1)

    # ==========================================================================
    def find_cell_boundary(self):
        """ line_image를 기준으로 각 Cell의 중심 좌표에서 상하좌우로 흰색(b,g,r = 255)값이 있는지 판별합니다.
        만약 흰색값이 있다면 경계(boundary)가 있는 것으로 판별합니다.
        """
        for cols in range(len(self.final_y) - 1):
            for rows in range(len(self.final_x) - 1):
                x, y, width, height, central_x, central_y = self.cells[cols][rows]._get_value()
                if rows == 0:
                    self.cells[cols][rows].boundary['left'] = True
                if rows == len(self.final_x) - 1:
                    self.cells[cols][rows].boundary['right'] = True
                if cols == 0:
                    self.cells[cols][rows].boundary['upper'] = True
                if cols == len(self.final_y) - 1:
                    self.cells[cols][rows].boundary['lower'] = True

                # 'white'의 b != 0
                for left in range(x + 1, central_x):
                    if self.line_image.item(central_y, left, 0) != 0:
                        self.cells[cols][rows].boundary['left'] = True
                        break
                for right in range(x + width - 1, central_x, -1):
                    if self.line_image.item(central_y, right, 0) != 0:
                        self.cells[cols][rows].boundary['right'] = True
                        break
                for upper in range(y + 1, central_y):
                    if self.line_image.item(upper, central_x, 0) != 0:
                        self.cells[cols][rows].boundary['upper'] = True
                        break
                for lower in range(y + height - 1, central_y, -1):
                    if self.line_image.item(lower, central_x, 0) != 0:
                        self.cells[cols][rows].boundary['lower'] = True
                        break

    # ==========================================================================
    def merge_cell(self):
        self.merge_cell_horizontal()
        self.merge_cell_vertical()

        if self.verbose.startswith('vv'):
            tmp_img = np.copy(self.erased_line)

            cols_range = len(self.cells)
            for cols in range(cols_range):
                rows_range = len(self.cells[cols])
                for rows in range(rows_range):
                    x, y, width, height, _, _ = self.cells[cols][rows]._get_value()
                    cv2.rectangle(tmp_img, (x, y), (x + width, y + height),
                                  (255, 0, 0, 50), 2)

            self._show_img('_cell_merged', tmp_img)

    # ==========================================================================
    def merge_cell_horizontal(self):
        """ 좌우로 열거된 Cell들에 대해 merge 작업을 수행합니다.
        만약 현재 Cell의 우측 경계가 있거나, 우측 Cell의 좌측 경계가 있다면 경계가 있는 것으로 인식하고,
        그렇지 않다면 경계가 없는 것으로 인식, merge 작업을 수행합니다.
        """
        cols_range = len(self.cells)
        for cols in range(cols_range):
            rows_range = len(self.cells[cols])
            rows_flag = 0
            for rows in range(rows_range - 1):
                rows -= rows_flag

                now_b = self.cells[cols][rows].boundary
                right_b = self.cells[cols][rows + 1].boundary

                # todo 경계가 명확하지 않은 Cell에 대해 좌우 양 끝단에 있는 Cell을 나누어 줘야함
                if now_b['right'] or right_b['left']:
                    # 오른쪽 경계 있음
                    continue
                else:
                    if (now_b['upper'] == right_b['upper']) or (now_b['lower'] == right_b['lower']):
                        # merge horizontal
                        self.cells[cols][rows]._merge_horizontal(self.cells[cols][rows + 1])
                        del self.cells[cols][rows + 1]
                        rows_flag += 1
                        rows_range -= 1

                    else:
                        continue

                ''' 이건 셀 안의 셀의 경계를 나눠버림....
                                elif ((not now_b['lower']) and right_b['lower']) or (now_b['lower'] and (not right_b['lower'])):
                                    # 아래 경계가 없으면서 우측 셀은 아래 경계를 지닐 때
                                    continue
                                elif ((not now_b['upper']) and right_b['upper']) or (now_b['upper'] and (not right_b['upper'])):
                                    # 위 경계가 없으면서 우측 셀은 위 경계를 지닐 때
                                    continue
                                '''

    # ==========================================================================
    def merge_cell_vertical(self):
        """ 상하로 열거된 Cell들에 대해 merge 작업을 수행합니다.
        만약 현재 Cell의 하단 경계가 있거나, 하단 Cell의 상단 경계가 있다면 경계가 있는 것으로 인식하고,
        그렇지 않다면 경계가 없는 것으로 인식, merge 작업을 수행합니다.
        """
        # todo 현재 merge algorithm(알고리즘)은 merge_horizontal 이후 인덱스 다 꼬이게 됨
        # 고민해서 더 나은 algorithm(알고리즘)을 설계해야함

        for cols in range(len(self.cells)):
            for rows in range(len(self.cells[cols])):
                now = self.cells[cols][rows]
                merge_flag = False

                for tmp_col in range(cols + 1, len(self.cells)):
                    for tmp_row in range(len(self.cells[tmp_col])):
                        if tmp_col == cols + 1 or (tmp_col != cols + 1 and merge_flag):
                            tmp_cell = self.cells[tmp_col][tmp_row]
                            if (now.x == tmp_cell.x) and (now.width == tmp_cell.width) and (
                                    now.y + now.height == tmp_cell.y):
                                if now.boundary['lower'] or tmp_cell.boundary['upper']:
                                    continue
                                else:
                                    now._merge_vertical(tmp_cell)
                                    del self.cells[tmp_col][tmp_row]
                                    merge_flag = True
                                    break
                            else:
                                continue
                    if tmp_col == cols + 1 and not merge_flag:
                        break

    # ==========================================================================
    def process(self):
        self.boxing_ambiguous()
        self.detect_contours()
        self.morph_closing()
        self.erase_line()
        # self.detect_line()
        self.cal_cell_needed()
        self.save_cell_value()
        self.find_cell_boundary()
        self.merge_cell()

    # ==========================================================================
    def temp_print(self):
        """ cell's info printing method for debugging
        """
        for cols in range(len(self.cells)):
            for rows in range(len(self.cells[cols])):
                print(self.cells[cols][rows])


################################################################################
class OcrByCell(Preprocessing):
    def __init__(self, img_file, conf_file=None, verbose='vv'):
        Preprocessing.__init__(self, img_file, conf_file=conf_file, verbose=verbose)

    # ==========================================================================
    def ocr_by_box(self):
        i = 0
        for cols in range(len(self.cells)):
            for rows in range(len(self.cells[cols])):
                x, y, width, height, _, _ = self.cells[cols][rows]._get_value()
                # separated_img가 원본이미지를 벗어나면 안된다
                if cols == 0 or rows == 0 or cols == len(self.cells) or rows == len(self.cells[cols]):
                    separated_img = self.erased_line[y:y + height, x:x + width]
                else:
                    separated_img = self.erased_line[y - 3:y + height + 3, x - 3:x + width + 3]

                img_for_calculate = self.get_processing_img(separated_img)
                # 분리된 이미지에서 글자영역을 찾아 높이를 계산, 폰트 사이즈를 유추.
                self.cells[cols][rows].text_height = self.get_text_height(img_for_calculate)

                # 분리된 이미지에서 글자영역을 찾아 align, valign을 유추
                self.cells[cols][rows].text_align, self.cells[cols][rows].text_valign \
                    = self.get_text_align(img_for_calculate)

                # 글자영역을 지우고 남은 색상으로 셀 배경색 추출
                self.cells[cols][rows].bg_color = self.get_bg_color(separated_img, img_for_calculate)

                # 잔여 line 및 잡영 제거
                separated_img = self.erase_line_and_noise(separated_img)

                separated_img = self.zoom_image(separated_img)
                separated_img = self.add_white_space(separated_img)
                '''
                Page segmentation modes:
                      0    Orientation and script detection (OSD) only.
                      1    Automatic page segmentation with OSD.
                      2    Automatic page segmentation, but no OSD, or OCR.
                      3    Fully automatic page segmentation, but no OSD. (Default)
                      4    Assume a single column of text of variable sizes.
                      5    Assume a single uniform block of vertically aligned text.
                      6    Assume a single uniform block of text.
                      7    Treat the image as a single text line.
                      8    Treat the image as a single word.
                      9    Treat the image as a single word in a circle.
                     10    Treat the image as a single character.
                     11    Sparse text. Find as much text as possible in no particular order.
                     12    Sparse text with OSD.
                     13    Raw line. Treat the image as a single text line,
                           bypassing hacks that are Tesseract-specific.
                '''
                text = ocr.image_to_string(separated_img, lang='kor+eng', config='psm 1')

                if not len(text):
                    pass
                else:
                    self.cells[cols][rows].text = text
                    length = len(self.before_merged[cols])
                    for i in range(0, length):
                        temp = self.before_merged[cols][i].central_x
                        if (temp > x and temp < x + width):
                            self.before_merged[cols][i].text = text
                    i += 1
                ''' 구두점...
                    if not len(text) or (ord(text[0]) < 48 and ord(text[0]) > 32) or (
                        ord(text[0]) < 65 and ord(text[0]) > 57) or (ord(text[0]) < 97 and ord(text[0]) > 90) or (
                        ord(text[0]) < 127 and ord(text[0]) > 122):
                        pass
                '''

        # self.temp_print()

    # ==========================================================================
    def add_white_space(self, img):
        """ OCR 인식률을 높이기 위해 여백을 줍니다.
        만약 분리되고 확대된 이미지가 Orignal_image 의 width, height 보다 크다면
        default값의 여백만 추가해 줍니다.
        :param img: Cell 영역별로 분리되고 확대된 image
        :return: 여백을 추가한 image를 return
        """
        # OCR 인식률을 높이기 위한 여백주기
        h = img.shape[0]
        w = img.shape[1]

        additional_w = self.config['improve_ocr']['additional_width']
        additional_h = self.config['improve_ocr']['additional_height']

        if self.Origin_image.shape[0] > h:
            additional_h = int((self.Origin_image.shape[0] - h) / 2)
        if self.Origin_image.shape[1] > w:
            additional_w = int((self.Origin_image.shape[1] - w) / 2)

        WHITE = [255, 255, 255]
        img = cv2.copyMakeBorder(img, additional_h, additional_h, additional_w,
                                 additional_w, cv2.BORDER_CONSTANT, value=WHITE)

        return img

    # ==========================================================================
    def zoom_image(self, img):
        """ OCR 인식률을 높이기 위해 약간의 확대작업을 수행합니다.
        config.yaml 에서 확대비율을 수정할 수 있습니다.
        :param img: Cell 영역별로 분리된 image
        :return: 확대된 image를 return
        """
        zoom_fx = self.config['improve_ocr']['zoom_fx']
        zoom_fy = self.config['improve_ocr']['zoom_fy']

        img = cv2.resize(img, None, fx=zoom_fx, fy=zoom_fy, interpolation=cv2.INTER_CUBIC)

        return img

    # ==========================================================================
    def get_processing_img(self, img):
        """ 글자영역을 제거하기 위해
        1) Gray-scale 적용
        2) Canny edge 추출 알고리즘 적용
        3) GaussianBlur 적용
        4) dilation 적용
        5) opening 적용
        :param img: Cell 영역별로 분리된 image
        :return: processsed image
        """
        temp_img = img
        imgray = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)

        low_threshold = self.config['canny']['low_threshold']
        high_threshold = self.config['canny']['high_threshold']
        edges = cv2.Canny(imgray, low_threshold, high_threshold)

        blur = cv2.GaussianBlur(edges, (3, 3), 0)

        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(blur, kernel, iterations=1)

        opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel, iterations=2)
        return opening

    # ==========================================================================
    def get_text_height(self, processed_img):
        """ 글자영역의 contour(윤곽)을 추출하여 높이에 따른 font-size를 유추합니다.
        10px == 7.5pt로 대략적으로 0.75배 처리.
        :param processed_img: 글자영역을 추출하기 위해 처리된 image
        :return: 유추된 font-size 만약 글자영역을 못찾았다면 defalut font-size를 반환
        """
        retrieve_mode = self.config['contour']['retrieve_mode']
        approx_method = self.config['contour']['approx_method']
        _, contours, _ = cv2.findContours(processed_img, retrieve_mode, approx_method)

        num = 0
        average_h = 0
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if (width > 10 and height > 10) and height < processed_img.shape[0] * 0.8:
                average_h += height
                num += 1
        if num:
            return int((average_h / num) * 0.75)  # 10px == 7.5pt
        else:
            return 11

    # ==========================================================================
    def detect_blank(self, processed_img):
        """ 글자영역을 기준으로 상하좌우 여백의 size를 구하여 반환합니다.
        :param processed_img: 글자영역을 추출하기 위해 처리된 image
        :return: 상하좌우의 여백길이
        """
        # todo 최소값을 반환하기 때문에 줄글로 이어져있는 텍스트(text)에 부적합, 개선된 알고리즘 필요
        retrieve_mode = self.config['contour']['retrieve_mode']
        approx_method = self.config['contour']['approx_method']
        _, contours, _ = cv2.findContours(processed_img, retrieve_mode, approx_method)

        origin_h = processed_img.shape[0]
        origin_w = processed_img.shape[1]

        upper_blank = 1000
        below_blank = 1000
        left_blank = 1000
        right_blank = 1000

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width > 10 and height > 10:
                contour_upper = y
                contour_below = origin_h - (y + height)
                contour_left = x
                contour_right = origin_w - (x + width)

                if upper_blank > contour_upper:
                    upper_blank = contour_upper
                if below_blank > contour_below:
                    below_blank = contour_below
                if left_blank > contour_left:
                    left_blank = contour_left
                if right_blank > contour_right:
                    right_blank = contour_right

        return upper_blank, below_blank, left_blank, right_blank

    # ==========================================================================
    def get_text_align(self, processed_img):
        """ Cell 내부의 텍스트(text)에 대해 좌우 정렬과 상하 정렬을 유추하여 반환합니다.
        :param processed_img: 글자영역을 추출하기 위해 처리된 image
        :return: 유추된 상하, 좌우의 text align
        """
        upper_blank, below_blank, left_blank, right_blank = self.detect_blank(processed_img)

        align = 'center'
        valign = 'vcenter'
        if max(upper_blank, below_blank) > min(upper_blank, below_blank) * 2:
            if min(upper_blank, below_blank) == upper_blank:
                # valign 상단 정렬
                valign = 'top'
            else:
                # valign 하단 정렬
                valign = 'bottom'
        if max(left_blank, right_blank) > min(left_blank, right_blank) * 2:
            if min(left_blank, right_blank) == left_blank:
                # align 좌측 정렬
                align = 'left'
            else:
                # align 우측 정렬
                align = 'right'

        return align, valign

    # ==========================================================================
    def get_bg_color(self, img, processed_img):
        """ Cell에서 글자영역을 찾아 흰색(b,g,r = 255)으로 지우고, Cell의 모든 픽셇의 값을 검사하여,
        배경셀의 색상을 유추합니다.
        :param img: 원본에서 분리된 이미지
        :param processed_img: 글자영역을 추출하기 위해 처리된 image
        :return: HEX COLOR로 변환한 색상값
        """
        temp_image = np.copy(img)
        retrieve_mode = self.config['contour']['retrieve_mode']
        approx_method = self.config['contour']['approx_method']
        _, contours, _ = cv2.findContours(processed_img, retrieve_mode, approx_method)

        origin_h = temp_image.shape[0]
        origin_w = temp_image.shape[1]

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            cv2.rectangle(temp_image, (x, y), (x + width, y + height), (255, 255, 255), cv2.FILLED)

        mean_b = 0
        mean_g = 0
        mean_r = 0
        tmp = 0

        ten_percent_h = int(origin_h * 0.1)
        ten_percent_w = int(origin_w * 0.1)

        for y in range(ten_percent_h, origin_h - ten_percent_h):
            for x in range(ten_percent_w, origin_w - ten_percent_w):
                if temp_image.item(y, x, 0) != 255 and temp_image.item(y, x, 1) != 255 \
                        and temp_image.item(y, x, 2) != 255:
                    mean_b += int(temp_image.item(y, x, 0))  # b
                    mean_g += int(temp_image.item(y, x, 1))  # g
                    mean_r += int(temp_image.item(y, x, 2))  # r
                    tmp += 1
        if not tmp:
            return '#ffffff'  # HEX 'WHITE'
        else:
            mean_b = int(np.ceil(mean_b / tmp))
            mean_g = int(np.ceil(mean_g / tmp))
            mean_r = int(np.ceil(mean_r / tmp))
            if mean_b > 245 and mean_g > 245 and mean_r > 245:
                return '#ffffff'

            else:
                # RGB to HEX
                return '#{:02x}{:02x}{:02x}'.format(mean_r, mean_g, mean_b)

    # ==========================================================================
    def erase_line_and_noise(self, img):
        """ HoughlineP 메소드를 이용하여 긴 line을 제거하고,
        Canny edge 추출 알고리즘을 이용하여 필요한 부분(text 영역)을 제외한 부분을 흰색으로 noise 를 지워낸다.
        :param img: 원본에서 Cell 영역별로 분리된 이미지
        :return: 긴 line과 노이즈를 지운 이미지
        """
        line_image = img * 0

        low_threshold = self.config['canny']['low_threshold']
        high_threshold = self.config['canny']['high_threshold']
        edges = cv2.Canny(img, low_threshold, high_threshold)

        rho = self.config['houghline']['rho']  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = self.config['houghline']['threshold']  # minimum number of votes (intersections in Hough grid Cell)
        min_line_length = min(img.shape[0], img.shape[1]) * 0.7  # minimum number of pixels making up a line
        max_line_gap = 0  # maximum gap in pixels between connectable line segments

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)
        dilation = cv2.dilate(closing, kernel, iterations=2)

        img = cv2.addWeighted(img, 1, line_image, 1, 0)
        dilation = cv2.cvtColor(~dilation, cv2.COLOR_GRAY2BGR)

        img = cv2.addWeighted(img, 1, dilation, 1, 0)

        return img

    # ==========================================================================
    '''
    def temp(self, img):
        ## 글자영역 추출 알고리즘
        temp_img = img

        imgray = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(imgray, 100, 200)

        blur = cv2.GaussianBlur(edges, (3, 3), 0)

        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(blur, kernel, iterations=2)

        opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel, iterations=2)

        _, contours, hierarchy = cv2.findContours(opening, cv2.RETR_CCOMP, 2)

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)

            if width > 15 and height > 15:
                opening = cv2.rectangle(opening, (x, y), (x + width, y + height), (255, 0, 0, 50), 1)

                cv2.imshow('contoursssss', opening)
                cv2.waitKey(0)

                separated_img = temp_img[y:y + height, x:x + width]

                separated_img = cv2.resize(separated_img, None, fx=2, fy=3, interpolation=cv2.INTER_CUBIC)

                WHITE = [255, 255, 255]
                separated_img = cv2.copyMakeBorder(separated_img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=WHITE)

                text = ocr.image_to_string(separated_img, lang='kor+eng')
                if not len(text):
                    pass
                else:
                    print('temp :\t', text)
                    cv2.imshow('tmp', separated_img)
                    cv2.waitKey(0)
    '''


################################################################################
class Export2Document(OcrByCell):
    def __init__(self, img_file, conf_file=None, verbose='vv', workbook='test.xlsx'):
        OcrByCell.__init__(self, img_file, conf_file=conf_file, verbose=verbose)
        # Create an new Excel file and add a worksheet.
        self.workbook = xlsxwriter.Workbook(workbook)

        self.default_format = dict()
        for key in self.config['xlsx_standard']['default_format']:
            self.default_format[key] = self.config['xlsx_standard']['default_format'][key]

    # ==========================================================================
    def export_to_xlsx(self):
        worksheet = self.workbook.add_worksheet()

        worksheet = self.make_base(worksheet)
        worksheet = self.merge_and_input_text(worksheet)

        self.workbook.close()

    # ==========================================================================
    def make_base(self, worksheet):
        """ 셀의 너비나 높이등 기본적인 틀을 구성합니다.
        :param worksheet: 작업할 worksheet
        :return: 작업후의 worksheet
        """
        default_format = self.workbook.add_format(self.default_format)
        # 인덱스가 다 꼬여있기 때문에 height도 뒤죽박죽, 머지전의 데이터가 필요
        for cols in range(0, len(self.before_merged)):
            worksheet.set_row(cols, self.before_merged[cols][0].height, default_format)  # height
            for rows in range(0, len(self.before_merged[cols])):
                present = self.before_merged[cols][rows]
                worksheet.set_column(rows, rows, present.width / 6.5, default_format)  # width/7
        return worksheet

    # ==========================================================================
    def merge_and_input_text(self, worksheet):
        """ Merge된 정보에 따라 Cell들을 Merge하고 OCR로 읽어낸 text를 입력합니다.
        만약, 현재 Cell의 이름과 merged_info의 이름이 같다면 Cell은 Merge되지 않았음을 의미
        그렇지 않다면 Cell이 Merge 되었음을 의미합니다.
        :param worksheet: 작업할 worksheet
        :return: 작업후의 worksheet
        """
        for cols in range(0, len(self.cells)):
            for rows in range(0, len(self.cells[cols])):
                present = self.cells[cols][rows]
                cell_format = self.get_text_attribute(present)
                cell_format.set_text_wrap()
                # not merged
                if present.cell_name == present.merged_info:
                    if not present.text:
                        worksheet.write_blank(present.cell_name, None, cell_format)
                    else:
                        worksheet.write_rich_string(present.cell_name, present.text, cell_format)
                # merged
                else:
                    worksheet.merge_range(present.cell_name + ':' + present.merged_info, None,
                                          cell_format=cell_format)
                    if not present.text:
                        worksheet.write_blank(present.cell_name, None, cell_format)
                    else:
                        worksheet.write_rich_string(present.cell_name, present.text, cell_format)

        return worksheet

    # ==========================================================================
    def get_text_attribute(self, cell):
        boundary = cell.boundary
        top = 0
        bottom = 0
        left = 0
        right = 0
        if boundary['upper']:  # if boundary['upper'] == True
            top = 1  # 1 == Continuous line
        if boundary['lower']:
            bottom = 1
        if boundary['left']:
            left = 1
        if boundary['right']:
            right = 1
        cell_format = self.workbook.add_format({'font_name': 'Calibri', 'font_color': '#000000',
                                                'align': cell.text_align, 'valign': cell.text_valign,
                                                'top': top, 'bottom': bottom, 'left': left, 'right': right,
                                                'font_size': cell.text_height, 'bg_color': cell.bg_color})
        return cell_format


################################################################################
def main():
    main_process = Export2Document('./data/FO-입사지원서-01.png', verbose='v')
    start_time = time.time()
    main_process.process()
    print(round(time.time() - start_time, 3), ' sec')

    start_time = time.time()
    main_process.ocr_by_box()
    # main_precess.simple_ocr()
    print(round(time.time() - start_time, 3), ' sec')

    start_time = time.time()
    # main_precess.write_to_txt()
    # main_precess.read_from_txt()
    main_process.export_to_xlsx()
    print(round(time.time() - start_time, 3), ' sec')

    main_process.temp_print()

    '''
    main_precess = Export2Document('./data/FO-계산서-01.png', verbose='v', workbook='계산서.xlsx')
    main_precess.process()
    main_precess.ocr_by_box()
    # main_precess.simple_ocr()
    # main_precess.write_to_txt()
    # main_precess.read_from_txt()
    main_precess.export_to_xlsx()

    main_precess = Export2Document('./data/FO-영수증-01.png', verbose='v', workbook='영수증.xlsx')
    main_precess.process()
    main_precess.ocr_by_box()
    main_precess.export_to_xlsx()

    main_precess = Export2Document('./data/FO-입사지원서-01.png', verbose='v', workbook='입사지원서.xlsx')
    main_precess.process()
    main_precess.ocr_by_box()
    main_precess.export_to_xlsx()

    main_precess = Export2Document('./data/FO-회원가입-01.png', verbose='v', workbook='회원가입.xlsx')
    main_precess.process()
    main_precess.ocr_by_box()
    main_precess.export_to_xlsx()
    '''


################################################################################
if __name__ == "__main__":
    main()
