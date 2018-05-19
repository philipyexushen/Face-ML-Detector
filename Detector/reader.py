#-*- coding: utf-8 -*-
from common import *
from lxml import etree
import os
import random

_maxCapture = 2000
_ImageSize = (64, 64, 3)
OutputSimpleFace = 0
OutputVOCData = 1

def GetOutputImageSize():
    return _ImageSize

def _DrawRects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def CreateVOCXml(path, fileName:str, shape, objectData):
    height, width, depth = shape

    annotation = etree.Element("annotation")
    etree.SubElement(annotation, "folder").text = "CUSTOM"
    etree.SubElement(annotation, "filename").text = fileName

    source = etree.SubElement(annotation, "source")
    etree.SubElement(source, "database").text = "CUSTOM"
    etree.SubElement(source, "annotation").text = "CUSTOM"
    etree.SubElement(source, "image").text = "CUSTOM"

    size = etree.SubElement(annotation, "size")
    etree.SubElement(size, "width").text = str(width)
    etree.SubElement(size, "height").text = str(height)
    etree.SubElement(size, "depth").text = str(depth)
    etree.SubElement(annotation, "segmented").text = '0'

    for obj in objectData:
        name, rect = obj
        x1, y1, x2, y2 = rect

        key_object = etree.SubElement(annotation, "object")
        etree.SubElement(key_object, "name").text = name

        bndbox = etree.SubElement(key_object, "bndbox")
        etree.SubElement(bndbox, "xmin").text = str(x1)
        etree.SubElement(bndbox, "ymin").text = str(y1)
        etree.SubElement(bndbox, "xmax").text = str(x2)
        etree.SubElement(bndbox, "ymax").text = str(y2)
        etree.SubElement(key_object, "difficult").text = "0"

    doc = etree.ElementTree(annotation)
    with open(os.path.join(path, "Annotations", f"{fileName[:-4]}.xml"), "wb") as f:
        doc.write(f, pretty_print=True)


def _StoreCap(img, rects, numCapture, strLabelName, outputType):
    if len(rects) == 0:
        return

    if outputType == OutputSimpleFace:
        for x1, y1, x2, y2 in rects:
            if x1 < 10 or x2 < 10 or y1 < 10 or y2 < 10:
                continue

            imgCap = img[y1 - 10:y2 + 10,x1 - 10: x2 + 10]
            cv.imwrite(f"./capture/{strLabelName}/{numCapture[0]}.jpg", imgCap)

            numCapture[0] += 1
            if numCapture[0] > _maxCapture:
                return

    elif outputType == OutputVOCData:
        # 只取面积最大的那个
        maxArea = 0
        bestRect = []
        for x1, y1, x2, y2 in rects:
            s = (x2 - x1) * (y2 - y1)
            # sb pycharm
            if x1 < 10 or x2 < 10 or y1 < 10 or y2 < 10:
                continue
            if s > maxArea:
                maxArea = s
                bestRect = [x1 - 10, y1 - 10, x2 + 10, y2 + 10]

        if len(bestRect) == 0:
            return

        fileName = f"{strLabelName}_{numCapture[0]}.jpg"
        CreateVOCXml(f"./capture/",fileName , img.shape, [[strLabelName, bestRect]])
        imgsets_path_trainval = os.path.join("./capture/", "ImageSets", "Main", "trainval.txt")
        imgsets_path_test = os.path.join("./capture/", 'ImageSets', 'Main', 'test.txt')
        bTrain = random.randint(0, 1)
        if bTrain == 1:
            with open(imgsets_path_trainval, "a") as f:
                f.writelines(f"{fileName[:-4]}\n")
        else:
            with open(imgsets_path_test, "a") as f:
                f.writelines(f"{fileName[:-4]}\n")

        cv.imwrite(os.path.join("./capture/JPEGImages", fileName), img)
        numCapture[0] += 1


def CaptureTrainingSet(outputType = OutputSimpleFace):
    strLabelName = str()
    print(os.curdir)
    while len(strLabelName) == 0:
        print("Please input data set label name")
        strLabelName = input()

    if outputType == OutputSimpleFace:
        if not os.path.isdir(f"./capture/{strLabelName}"): os.mkdir(f"./capture/{strLabelName}")

    if outputType == OutputVOCData:
        if not os.path.isdir(f"./capture/Annotations"): os.mkdir(f"./capture/Annotations")
        if not os.path.isdir(f"./capture/JPEGImages"): os.mkdir(f"./capture/JPEGImages")
        if not os.path.isdir(f"./capture/ImageSets"): os.mkdir(f"./capture/ImageSets")
        if not os.path.isdir(f"./capture/ImageSets/Main"): os.mkdir(f"./capture/ImageSets/Main")

        imgsets_path_trainval = os.path.join("./capture/", "ImageSets", "Main", "trainval.txt")
        imgsets_path_test = os.path.join("./capture/", 'ImageSets', 'Main', 'test.txt')

        if os.path.isfile(imgsets_path_trainval): os.remove(imgsets_path_trainval)
        if os.path.isfile(imgsets_path_test): os.remove(imgsets_path_test)

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

        t1 = Clock()
        rects = find_haar_rect(imgGray, cascade)
        _StoreCap(img, rects, numCapture, strLabelName, outputType)
        vis = img.copy()
        _DrawRects(vis, rects, (0, 255, 0))

        dt = Clock() - t1

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt * 1000))
        draw_str(vis, (20, 40), f"Have captured {numCapture[0]} faces")
        cv.imshow("Image Collector", vis)
        if cv.waitKey(5) == 27:
            break

    cap.release()
    cv.destroyAllWindows()


_NumOfLabel = 0

def _LoadDataInternal(filePath:str, tableImage:list, tableLabelIdx:list, dictNameLabel:dict, curLabelName:str):
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


