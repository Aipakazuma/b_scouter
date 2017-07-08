# -*- coding: utf8 -*-


import cv2
import numpy as np


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # 横サイズ
    cap.set(4, 480)  # 縦サイズ

    while(True):
        ret, frame = cap.read()
        if ret == False:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # グレースケールに変換

        cv2.imshow('fram', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.dstroyAllWindows()
