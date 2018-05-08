import lxml.etree as ET
import os
import shutil

_maxCount = 5000

def SelectSpecialItem(inputPath:str, outputPath:str, lstTypeName):
    data_paths = [os.path.join(inputPath, s) for s in ['VOC2007', 'VOC2012']]
    output_data_paths = [os.path.join(outputPath, s) for s in ['VOC2007', 'VOC2012']]
    itemName = []

    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)

    dictItemCount = { name : 0  for name in lstTypeName }

    print('Parsing annotation files')

    for data_path, data_output_path in zip(data_paths, output_data_paths):
        if not os.path.isdir(data_output_path):
            os.mkdir(data_output_path)

        annot_path = os.path.join(data_path, 'Annotations')
        imgs_path = os.path.join(data_path, 'JPEGImages')

        annot_path_output = os.path.join(data_output_path, 'Annotations')
        if not os.path.isdir(annot_path_output):
            os.mkdir(annot_path_output)

        imgs_path_output = os.path.join(data_output_path, 'JPEGImages')
        if not os.path.isdir(imgs_path_output):
            os.mkdir(imgs_path_output)

        imgsets_path_trainval = os.path.join(data_path, 'ImageSets', 'Main', 'trainval.txt')
        imgsets_path_test = os.path.join(data_path, 'ImageSets', 'Main', 'test.txt')

        trainval_fileName = []
        test_fileName = []
        try:
            with open(imgsets_path_trainval) as f:
                for line in f:
                    trainval_fileName.append(line.strip())
        except Exception as e:
            print(e)

        try:
            with open(imgsets_path_test) as f:
                for line in f:
                    test_fileName.append(line.strip())
        except Exception as e:
            if data_path[-7:] == 'VOC2012':
                # this is expected, most pascal voc distibutions dont have the test.txt file
                pass
            else:
                print(e)

        idx = 0
        for strAnnotName in os.listdir(annot_path):
            annot = os.path.join(annot_path, strAnnotName)
            try:
                idx += 1
                et = ET.parse(annot)
                element = et.getroot()

                element_objs = element.findall('object')
                element_filename = element.find('filename').text

                bInList = False
                for element_obj in element_objs:
                    class_name = element_obj.find('name').text
                    if class_name in lstTypeName and dictItemCount[class_name] < _maxCount:
                        dictItemCount[class_name] += 1
                        bInList = True
                        break

                if bInList and strAnnotName.endswith(".xml"):

                    itemName.append(strAnnotName[:-4])
                    shutil.copy(annot, annot_path_output)
                    strPicPath = os.path.join(imgs_path, element_filename)
                    shutil.copy(strPicPath, imgs_path_output)


            except Exception as e:
                print(annot, e)
                continue

        try:
            image_sets_output_path = os.path.join(data_output_path, 'ImageSets')
            if not os.path.isdir(image_sets_output_path):
                os.mkdir(image_sets_output_path)

            print("Begin write test.txt and trainval.txt")
            imageSetsMain_output_path = os.path.join(image_sets_output_path, 'Main')
            if not os.path.isdir(imageSetsMain_output_path):
                os.mkdir(imageSetsMain_output_path)

            test_file_output_path = os.path.join(imageSetsMain_output_path, "test.txt")
            trainval_file_output_Path = os.path.join(imageSetsMain_output_path, "trainval.txt")

            print(f"test size = {len(test_fileName)}")
            print(f"trainval size = {len(trainval_fileName)}")

            with open(test_file_output_path, "w") as fTest, open(trainval_file_output_Path, "w") as fTrainval:
                for name in itemName:
                    if name in trainval_fileName:
                        fTrainval.writelines(name + "\n")
                    elif name in test_fileName:
                        fTest.writelines(name + "\n")
            print(f"{test_file_output_path} and {trainval_file_output_Path} has been created!")
        except Exception as e:
            print(e)
            continue


inputPath = "G:/mldata/VOCdevkit/"
outputPath = "./VOCdevkit/"

SelectSpecialItem(inputPath, outputPath, ["dog", "cat", "person", "car", "horse"])