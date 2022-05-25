import torch
import cv2 as cv
import os
import numpy as np


if __name__ == '__main__':
    # 读取图像
    images = []
    for filename in os.listdir(r"../../source/train"):
        img = cv.imread("./train"+"/"+filename)
        # 转灰度
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # 高斯模糊
        # img = cv.GaussianBlur(img, (5, 5), 10)
        # 边缘检测
        # img = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=1)
        # img = np.power(img, 0.5)
        # img = cv.Canny(img, 250, 100)
        kernel_x = np.array([[1, 0], [0, -1]], dtype=int)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=int)
        x = cv.filter2D(img, cv.CV_16S, kernel_x)
        y = cv.filter2D(img, cv.CV_16S, kernel_y)
        absx = cv.convertScaleAbs(x)
        absy = cv.convertScaleAbs(y)
        img = cv.addWeighted(absx, 0.5, absy, 0.5, 0)
        # 二值化
        ret, img = cv.threshold(img, 32, 255, cv.THRESH_BINARY)
        # 腐蚀
        kernel = np.ones((3, 3), np.uint8)
        img = cv.erode(img, kernel, iterations=1)
        # # 轮廓查找
        # i, j = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # for i1 in i:
        #     x, y, w, h = cv.boundingRect(i1)
        #     if w > 2*h:
        #         img = img[y:y+h][x:x+w]
        # # gamma校正
        # img = np.power(img, 0.5)

        images.append(img)
    # print(images)
    cv.imshow('image', images[0])
    cv.waitKey()
