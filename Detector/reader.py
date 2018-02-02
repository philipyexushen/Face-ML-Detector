#-*- coding: utf-8 -*-
from common import *

_maxCapture = 4000
_ImageSize = (64, 64, 3)

def GetOutputImageSize():
    return _ImageSize

def _Clock():
    return cv.getTickCount() / cv.getTickFrequency()


def FindHaarRect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects


def _DrawRects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def _StoreCap(img, rects, numCapture, strLabelName):
    if len(rects) > 0:
        for x1, y1, x2, y2 in rects:
            if x1 < 10 or x2 < 10 or y1 < 10 or y2 < 10:
                continue

            imgCap = img[y1 - 10:y2 + 10,x1 - 10: x2 + 10]
            cv.imwrite(f"./capture/{strLabelName}/{numCapture[0]}.jpg", imgCap)

            numCapture[0] += 1
            if numCapture[0] > _maxCapture:
                return


def drawStr(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def CaptureTrainingSet():
    strLabelName = str()
    while len(strLabelName) == 0:
        print("Please input data set label name")
        strLabelName = input()

    cv.namedWindow("Image Collector")
    cascade = cv.CascadeClassifier("./haar_detector/haarcascade_frontalface_alt2.xml")
    cap = cv.VideoCapture(cameraIdx)

    if cap is None or not cap.isOpened():
        print("video can not open")
        exit(-1)

    numCapture = [0]
    while cap.isOpened() and numCapture[0] <= _maxCapture:
        _, img = cap.read()
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        imgGray = cv.equalizeHist(imgGray)

        t1 = _Clock()
        rects = FindHaarRect(imgGray, cascade)
        _StoreCap(img, rects, numCapture, strLabelName)
        vis = img.copy()
        _DrawRects(vis, rects, (0, 255, 0))

        dt = _Clock() - t1

        drawStr(vis, (20, 20), 'time: %.1f ms' % (dt * 1000))
        drawStr(vis, (20, 40), f"Have captured {numCapture[0]} faces")
        cv.imshow("Image Collector", vis)
        if cv.waitKey(5) == 27:
            break

    cap.release()
    cv.destroyAllWindows()


_NumOfLabel = 0

def _LoadDataInternal(filePath: str, tableImage:list, tableLabelIdx:list, dictNameLabel:dict, curLabelName: str):
    for itemName in os.listdir(filePath):
        absItemPath = os.path.abspath(os.path.join(filePath, itemName))

        if os.path.isdir(absItemPath):
            _LoadDataInternal(absItemPath, tableImage, tableLabelIdx, dictNameLabel, itemName)
        elif itemName.endswith(".jpg"):
            img = cv.imread(absItemPath)

            img = cv.resize(img, _ImageSize[:2])

            tableImage.append(img)
            if curLabelName in dictNameLabel:
                tableLabelIdx.append(dictNameLabel[curLabelName])
            else:
                global _NumOfLabel
                dictNameLabel.update({curLabelName : _NumOfLabel})
                tableLabelIdx.append(dictNameLabel[curLabelName])
                _NumOfLabel += 1


def LoadData(filePath: str = "./capture"):
    global _NumOfLabel
    _NumOfLabel = 0

    tableImage = list()
    tableLabelIdx = list()
    dictNameLabel = dict()
    _LoadDataInternal(filePath, tableImage, tableLabelIdx, dictNameLabel, "_")

    return np.array(tableImage), np.array(tableLabelIdx), _NumOfLabel, dictNameLabel


