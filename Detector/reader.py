#-*- coding: utf-8 -*-
import cv2.cv2 as cv

_cameraIdx = 0
_maxCapture = 4000

def _clock():
    return cv.getTickCount() / cv.getTickFrequency()


def _Detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects


def _DrawRects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def _StoreCap(img, rects, numCapture):
    if len(rects) > 0:
        for x1, y1, x2, y2 in rects: 
            imgCap = img[y1 - 10:y2 + 10,x1 - 10: x2 + 10]
            cv.imwrite(f"./capture/{numCapture[0]}.jpg", imgCap)

            numCapture[0] += 1
            if numCapture[0] > _maxCapture:
                return


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def CaptureTrainingSet():
    cv.namedWindow("Image Collector")
    cascade = cv.CascadeClassifier("./haar_detector/haarcascade_frontalface_alt2.xml")
    cap = cv.VideoCapture(_cameraIdx)

    if cap is None or not cap.isOpened():
        print("video can not open")
        exit(-1)

    numCapture = [0]
    while cap.isOpened() and numCapture[0] <= _maxCapture:
        ret, img = cap.read()
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        imgGray = cv.equalizeHist(imgGray)

        t1 = _clock()
        rects = _Detect(imgGray, cascade)
        _StoreCap(img, rects, numCapture)
        vis = img.copy()
        _DrawRects(vis, rects, (0, 255, 0))

        dt = _clock() - t1

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt * 1000))
        draw_str(vis, (20, 40), f"Have captured {numCapture[0]} faces")
        cv.imshow("Image Collector", vis)
        if cv.waitKey(5) == 27:
            break

    cap.release()
    cv.destroyAllWindows()


