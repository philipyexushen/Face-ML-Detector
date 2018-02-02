#-*- coding: utf-8 -*-
import numpy as np
import os
from numba import jit
import cv2 as cv
import matplotlib.pyplot as plt

cameraIdx = 0

def MethodInformProvider(method):
    def _decorator(*args, **kwargs):
        import time
        t0 = time.clock()
        print(f"[Call {method.__name__}]")
        ret = method(*args, **kwargs)
        t1 = time.clock()
        print(f"[Method {method.__name__} take {t1 - t0}s to execute]")
        print(f"[Out {method.__name__}]")
        return ret
    return _decorator


def Clock():
    return cv.getTickCount() / cv.getTickFrequency()


def DrawStr(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def FindHaarRect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects