#-*- coding: utf-8 -*-
from common import *
from reader import *
import keras
from CNNnetwork import *

_winName = "Shrimp Dumpling Classifier"

def HaarDetector(model:Model, dataSet = None):
    def _CreateIdx2NameMap(dictNameLabel:dict):
        dictIdx2NameMap = dict()
        for key, value in dictNameLabel.items():
            dictIdx2NameMap.update({value:key})
        return dictIdx2NameMap

    if dataSet is None:
        dataSet = model.dataSet

    cap = cv.VideoCapture(cameraIdx)
    cv.namedWindow(_winName)
    cascade = cv.CascadeClassifier("./haar_detector/haarcascade_frontalface_alt2.xml")
    dictIdx2NameMap = _CreateIdx2NameMap(dataSet.dictNameLabel)

    while cap.isOpened():
        _, imgSrc = cap.read()
        imgGray = cv.cvtColor(imgSrc, cv.COLOR_BGR2GRAY)
        imgGray = cv.equalizeHist(imgGray)

        rects = FindHaarRect(imgGray, cascade)
        if len(rects) > 0:
            for x1, y1, x2, y2 in rects:
                if x1 < 10 or x2 < 10 or y1 < 10 or y2 < 10:
                    continue

                imgCap = imgSrc[y1 - 10: y2 + 10, x1 - 10: x2 + 10]
                accuracy, resultLabelIdx = model.DetectFace(imgCap)
                cv.rectangle(imgSrc, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(imgSrc, dictIdx2NameMap[resultLabelIdx], (x1, y1 + 10), cv.FONT_HERSHEY_PLAIN, 2,
                           (255, 0, 0), lineType=cv.LINE_AA, thickness=1)
                #cv.putText(imgSrc, f"accuracy = {accuracy[0]}", (x1, y1 + 30), cv.FONT_HERSHEY_PLAIN, 1.0,
                #           (255, 0, 0), lineType=cv.LINE_AA)

        cv.imshow(_winName, imgSrc)
        if cv.waitKey(5) == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    print(keras.backend.image_dim_ordering())
    # CaptureTrainingSet()
    dataset = DataSet("./capture")
    dataset.Load()

    model = Model()
    model.Build(dataset)
    model.Train()
    model.SaveModel()
    # model = Model()
    # model.LoadModel()

    # begin test
    HaarDetector(model,dataset)

