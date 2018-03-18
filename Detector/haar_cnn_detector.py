#-*- coding: utf-8 -*-
from reader import *
from CNNnetwork import *
from common import *
import ffmpeg
import cv2.ocl as ocl

_winName = "Hargow Classifier"

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
        t1 = Clock()
        imgGray = cv.cvtColor(imgSrc, cv.COLOR_BGR2GRAY)
        imgGray = cv.equalizeHist(imgGray)

        rects = FindHaarRect(imgGray, cascade)

        dt = Clock() - t1
        DrawStr(imgSrc, (20, 20), 'Haar detector time: %.1f ms' % (dt * 1000))

        t1 = Clock()
        if len(rects) > 0:
            for x1, y1, x2, y2 in rects:
                if x1 < 10 or x2 < 10 or y1 < 10 or y2 < 10:
                    continue

                imgCap = imgSrc[y1 - 10: y2 + 10, x1 - 10: x2 + 10]
                accuracy, resultLabelIdx = model.DetectFace(imgCap)
                cv.rectangle(imgSrc, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(imgSrc, dictIdx2NameMap[resultLabelIdx], (x1, y1 + 20), cv.FONT_HERSHEY_SIMPLEX, 1.5,
                           (255, 0, 255), lineType=cv.LINE_AA, thickness = 2)

        dt = Clock() - t1
        DrawStr(imgSrc, (20, 40), 'Cnn detector time: %.1f ms' % (dt * 1000))
        cv.imshow(_winName, imgSrc)
        if cv.waitKey(5) == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    ocl.setUseOpenCL(True)
    # print(keras.backend.image_dim_ordering())
    '''
    CaptureTrainingSet(outputType=OutputVOCData)
    '''
    '''
        cap = cv.VideoCapture(0)
    index = 0
    cv.namedWindow("hello world")
    while cap.isOpened():
        _, img = cap.read()

        cv.imshow("hello world", img)
        cv.waitKey(5)
        cv.imwrite(f"E:\\Users\\Administrator\\pictures\\Test\\cap\\{index}.jpg", img)
        index += 1
    '''

    dataset = DataSet("./capture")
    dataset.Load()
    # model = Model()
    #model.Build(dataset)
    #model.Train()
    #model.SaveModel()
    model = Model()
    model.LoadModel()

    # begin test
    HaarDetector(model,dataset)

