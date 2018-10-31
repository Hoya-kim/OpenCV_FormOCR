import pytesseract as ocr
from PIL import Image
import cv2
import numpy as np
import pylab
import csv


def image_to_text():
    # 0. BASE
    filename = '../data/result/boxing_clarify.png'
    # image = Image.open(filename)
    # image = cv2.imread(filename) #어떻게 불러오든 결과값은 똑같음

    # 1. 그레이스케일 --> 오히려 인식률 떨어짐
    # image = cv2.imread(filename)
    # imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # 2. image_to_boxes --> 실패 macOS 지원 안 하는 듯?
    # text = image_to_boxes(image, lang='kor')
    # text2 = image_to_string(np.asarray(image), lang='kor')

    # 3. 라인 지운 이미지 --> 일반 이미지보다 인식률 좋음 *** -->라인없이 OCR해야함
    # filename = './data/result/erase_lines.png'
    image = cv2.imread(filename)

    # 4. Contour로 박스인식 후 박스단위로 OCR을 돌려보자

    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    '''
    # read the image and get the dimensions
    h, w, _ = image.shape  # assumes color image
    
    # run tesseract, returning the bounding boxes
    boxes = ocr.image_to_boxes(image)  # also include any config options you use

    # draw the bounding boxes on the image
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    # show annotated image and wait for keypress
    cv2.imshow(filename, image)
    cv2.waitKey(0)
    '''

    ret, thr = cv2.threshold(imgray, 230, 255, 0)

    _, contours, hierarchy = cv2.findContours(thr, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    min_width = 15
    min_height = 15

    i = 0

    for contour in contours:

        # epsilon = 0.1 * cv2.arcLength(contour, True)
        # approx = cv2.approxPolyDP(contour, epsilon, True)

        # top-left vertex coordinates (x,y) , width, height
        x, y, width, height = cv2.boundingRect(contour)
        # Draw screenshot that are larger than the standard size
        if width > 30 or height > 30:
            temp_img = (image[y-1:y + height+1, x-1:x + width+1])

            text = ocr.image_to_string(temp_img, lang='kor')

            if not len(text):
                pass
            else:
                print(text)
                cv2.imshow(str(i), temp_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # 중복으로 OCR 되는 것을 막기 위해 하얀색으로 채워줌
            image = cv2.rectangle(image, (x, y), (x + width, y + height),
                                  (255, 0, 0, 50), 2)
            cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), cv2.FILLED)
    i += 1

    cv2.imshow('aa',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # read the image and get the dimensions
    img = cv2.imread('./data/ocr_test_kor.png')
    h, w, _ = img.shape  # assumes color image

    # run tesseract, returning the bounding boxes
    boxes = pytesseract.image_to_boxes(img, 'kor')  # also include any config options you use

    # draw the bounding boxes on the image
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    # show annotated image and wait for keypress
    cv2.imshow('res', img)
    cv2.waitKey(0)
    '''

    '''
    pytesseract.run_tesseract('bw.png', 'output', lang=None, boxes=True, config="hocr")

    # To read the coordinates
    boxes = []
    with open('output.box', 'rb') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if (len(row) == 6):
                boxes.append(row)

    # Draw the bounding box
    img = cv2.imread('./data/ocr_test_kor.png')
    h, w, _ = img.shape
    for b in boxes:
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (255, 0, 0), 2)

    cv2.imshow('output', img)
    '''


image_to_text()
