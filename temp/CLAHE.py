import cv2
import numpy as np
from matplotlib import pyplot as plt

def CLAHE():
    img = cv2.imread('../data/FO-영수증-01.png',0);

    # contrast limit가 2이고 tile의 size는 8X8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img2 = clahe.apply(img)

    dst = np.hstack((img, img2))
    cv2.imshow('img',dst)

    # Bilateral Filtering
    dst = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imshow('img2', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


CLAHE()