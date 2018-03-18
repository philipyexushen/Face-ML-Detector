#-*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from reader import *
import random
import keras
from keras.utils import np_utils
from keras.optimizers import SGD
from common import *

_MODEL_PATH = './faceModel.h5'

class DataSet:
    def __init__(self, pathName):
        self.trainImages:np.ndarray = None
        self.trainLabels:np.ndarray = None

        self.validImages:np.ndarray = None
        self.validLabels:np.ndarray = None

        self.testImages:np.ndarray = None
        self.testLabels:np.ndarray = None

        self.pathName:str = pathName
        self.inputShape:tuple = None

        self.dictNameLabel:set = None

        self.numClasses:int = 0

    def Load(self):
        imgRows, imgCols, imgChannels = GetOutputImageSize()
        images, labels, numClasses, dictNameLabel = LoadData(self.pathName)

        trainImages, validImages, trainLabels, validLabels = train_test_split(images, labels, test_size=0.3,
                                                                            random_state=random.randint(0, 100))
        _, testImages, _, testLabels = train_test_split(images, labels, test_size=0.5,
                                                        random_state=random.randint(0, 100))

        # 这部分代码就是根据keras库要求的维度顺序重组训练数据集
        trainImages = trainImages.reshape(trainImages.shape[0], imgRows, imgCols, imgChannels)
        validImages = validImages.reshape(validImages.shape[0], imgRows, imgCols, imgChannels)
        testImages = testImages.reshape(testImages.shape[0], imgRows, imgCols, imgChannels)
        self.inputShape = (imgRows, imgCols, imgChannels)

        # 输出训练集、验证集、测试集的数量
        print(trainImages.shape[0], 'train samples')
        print(validImages.shape[0], 'valid samples')
        print(testImages.shape[0], 'test samples')

        trainLabels = np_utils.to_categorical(trainLabels, numClasses)
        validLabels = np_utils.to_categorical(validLabels, numClasses)
        testLabels = np_utils.to_categorical(testLabels, numClasses)

        trainImages = trainImages.astype('float32')
        validImages = validImages.astype('float32')
        testImages = testImages.astype('float32')
        cv.normalize(trainImages, trainImages, 0, 1, cv.NORM_MINMAX)
        cv.normalize(validImages, validImages, 0, 1, cv.NORM_MINMAX)
        cv.normalize(testImages, testImages, 0, 1, cv.NORM_MINMAX)

        self.trainImages = trainImages
        self.validImages = validImages
        self.testImages = testImages
        self.trainLabels = trainLabels
        self.validLabels = validLabels
        self.testLabels = testLabels
        self.numClasses = numClasses
        self.dictNameLabel = dictNameLabel


class Model:
    def __init__(self):
        self.model:Sequential = None
        self.dataSet = None

    def Build(self, dataSet:DataSet):
        self.model = Sequential()

        self.model.add(Convolution2D(32, kernel_size=(3, 3), padding="same", input_shape=dataSet.inputShape))
        self.model.add(Activation("relu"))

        self.model.add(Convolution2D(32, kernel_size=(3, 3)))
        self.model.add(Activation("relu"))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(64, kernel_size=(3, 3), padding="same"))
        self.model.add(Activation("relu"))

        self.model.add(Convolution2D(64, kernel_size=(3, 3)))
        self.model.add(Activation("relu"))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(dataSet.numClasses))
        self.model.add(Activation("softmax"))

        self.model.summary()
        self.dataSet = dataSet

    @MethodInformProvider
    def Train(self):
        # Stochastic gradient descent optimizer随机梯度下降
        sgdOptimizers = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
        self.model.compile(optimizer=sgdOptimizers, loss="categorical_crossentropy", metrics=['accuracy'])

        # 数据提升
        dataGen = ImageDataGenerator(featurewise_center=False,
                                        samplewise_center=False,
                                        featurewise_std_normalization=False,
                                        samplewise_std_normalization=False,
                                        zca_whitening=False,
                                        zca_epsilon=1e-6,
                                        rotation_range=20.,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        channel_shift_range=0.2,
                                        fill_mode='nearest',
                                        cval=0.,
                                        horizontal_flip=True,
                                        vertical_flip=False,
                                        rescale=None,
                                        preprocessing_function=None)

        dataGen.fit(self.dataSet.trainImages)
        self.model.fit_generator(dataGen.flow(self.dataSet.trainImages, self.dataSet.trainLabels,
                                              batch_size=20),
                                 epochs=2,
                                 steps_per_epoch=self.dataSet.trainImages.shape[0],
                                 validation_data=(self.dataSet.validImages, self.dataSet.validLabels))

    def SaveModel(self, path = _MODEL_PATH):
        self.model.save(path)

    def LoadModel(self, path = _MODEL_PATH):
        self.model = keras.models.load_model(path)

    def Evaluate(self):
        score = self.model.evaluate(self.dataSet.testImages, self.dataSet.testLabels)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    def DetectFace(self, imgSrc:np.ndarray):
        h, w, c = GetOutputImageSize()
        imgSrc:np.ndarray = cv.resize(imgSrc, (h, w))
        imgSrc = imgSrc.reshape(1, h, w, c)

        imgSrc = imgSrc.astype("float32")
        cv.normalize(imgSrc, imgSrc, 0, 1, cv.NORM_MINMAX)

        accuracy = self.model.predict_proba(imgSrc)
        resultClasses = self.model.predict_classes(imgSrc)

        return accuracy, resultClasses[0]