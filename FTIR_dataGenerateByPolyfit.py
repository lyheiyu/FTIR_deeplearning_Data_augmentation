from keras.layers import Dense, Input,add
from keras.models import Model
import numpy as np
import pandas  as  pd
import matplotlib.pyplot as plt
from sklearn.neural_network import  MLPClassifier
import  torch
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import keras as K
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from math import isnan
from sklearn.model_selection import cross_val_score
from sklearn import  svm
from sklearn.model_selection import train_test_split
from utils import utils
import tensorflow as tf
import random
import sys  # 导入sys模块
# sys.setrecursionlimit(30000)
from sklearn.preprocessing import normalize
MMScaler = MinMaxScaler()
class TestKeras:
    def __init__(self):
        pass
    def getPN(fileName):
        dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)

        polymerName = dataset.iloc[1:, 1]
        PN = []
        for item in polymerName:
            if item not in PN:
                PN.append(item)
        return PN

class TwodataAugmentation(object):
    def __init__(self, wavelength, x_train, y_train, polymerName):
        self.wavelength = wavelength
        self.x_train = x_train
        self.y_train = y_train
        self.polyerName = polymerName

    def generateData(self,num):
        numPoly = num
        ylabel = []
        data = []
        data2=[]
        data3=[]
        data4=[]
        if self.wavelength[0] > self.wavelength[-1]:
            rng = self.wavelength[0] - self.wavelength[-1]
        else:
            rng = self.wavelength[-1] - self.wavelength[0]
        half_rng = rng / 2
        normalized_wns = (self.wavelength - np.mean(self.wavelength)) / half_rng

        for m in range(len(self.x_train)):

            for n in range(len(self.x_train)):
                data.append([self.x_train[m],self.x_train[n]])
                data3.append(self.x_train[n])
                data2.append(self.x_train[m])

                rm=randomText(len(self.x_train), m, n)
                data4.append(self.x_train[rm])

                ylabel.append(numPoly)

            #data=np.array(data,dtype=np.float32)
            #print(data.shape)

        ylabel=np.array(ylabel)

        # data=np.array(data,dtype=float)
        # data2=np.array(data2,dtype=float)
        # data3=np.array(data3,dtype=float)
        # data4 = np.array(data4, dtype=float)
        # input1 = Input(shape=(1761,))
        # input2 = Input(shape=(1761,))
        # input3 = Input(shape=(1761,))
        #inputshape = len(self.x_train[0])
        data = np.array(data, dtype=float)
        data2 = np.array(data2, dtype=float)
        data3 = np.array(data3, dtype=float)
        data4 = np.array(data4, dtype=float)
        dataforpoly=[]
        dataforreturn=[]
        for i in range(len(data2)):
            datafortrain=(data[i][0]+data[i][1])/2
            dataforpoly.append(datafortrain)
            # model = K.Sequential()
            # model.add(Dense(units=1, activation='linear', input_shape=[1]))
            # model.add(Dense(units=128, activation='relu'))
            # model.add(Dense(units=256, activation='relu'))
            # model.add(Dense(units=300, activation='relu'))
            # model.add(Dense(units=256, activation='relu'))
            # model.add(Dense(units=128, activation='relu'))
            # model.add(Dense(units=1, activation='linear'))
            # model.compile(loss='mse', optimizer="adam")
            # model.summary()
            # model.fit(normalized_wns,datafortrain,epochs = 30,
            #                 batch_size = 512,
            #           verbose=True)
            # dataforadd=model.predict(normalized_wns)
            # dataforreturn.append(dataforadd)
        ## for tanh
        ## sigmoid
        # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


        #best batch_size=512,epochs=6

        return dataforreturn,ylabel,dataforpoly
def randomText(spectrumLength,m,n):
    length = spectrumLength
    if length < 1:
        return ''
    if length == 1:
        return 0
    if length == 2:
        return spectrumLength-1

    randomNumber= random.randint(0, length - 1)
    if randomNumber==m or randomNumber==n:
       return randomText(spectrumLength,m,n)

    return randomNumber

# if __name__ == '__main__':
#     td=TwodataAugmentation
#
#     from sklearn.model_selection import train_test_split
#
#     polymerName, waveLength, intensity, polymerID = utils.parseDataForSecondDataset('new_SecondDataset2.csv')
#     from FTIR_dataGenerateByConcentratedNet import TwoDimensionalAugmentation
#
#     cmTotal = np.zeros((4, 4))
#     m = 0
#     t_report = []
#     scoreTotal = np.zeros(5)
#     for seedNum in range(1):
#         x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=seedNum)
#         waveLength = np.array(waveLength, dtype=np.float)
#         datas = []
#         datas2 = []
#         PN = []
#         for item in polymerName:
#             if item not in PN:
#                 PN.append(item)
#         pID = []
#         for item in y_train:
#             if item not in pID:
#                 pID.append(item)
#         print(pID)
#         if len(pID) <= 3:
#             continue
#         for n in range(len(PN)):
#             numSynth = 2
#             indicesPS = [l for l, id in enumerate(y_train) if id == n]
#             intensityForLoop = x_train[indicesPS]
#             datas.append(intensityForLoop)
#             datas2.append(intensityForLoop)
#
#             augmentedSpectrum = []
#
#         for n in range(1, 3):
#
#             numSynth = 1
#             indicesPS = [i for i, id in enumerate(y_train) if id == n]
#             y_trainindex = y_train[indicesPS]
#             intensityForLoop = x_train[indicesPS]
#             referenceMean = np.mean(intensityForLoop, axis=0)
#
#
#
#             intensityplusnoise = []
#             intensityforRandomLoop = []
#             idexes = np.arange(len(intensityForLoop))
#
#             indexForIntensity = idexes.take(range(0, 200), mode='wrap')
#             intensityforRandomLoop = intensityForLoop[indexForIntensity]
#             # deirvesforloopindex = idexes.take(range(0, 200), mode='wrap')
#             # deirvesforloop = deirves[deirvesforloopindex]
#             # deirves2forloopindex = idexes.take(range(0, 200),mode='wrap')
#             # print(deirves2)
#             # print(deirves2forloopindex)
#             # deirves2forloop2=deirves2[deirves2forloopindex]
#             # deirvesforloopindex = idexes.take(range(0, 200), mode='wrap')
#             # deirvesforloop3 = deirves3[deirvesforloopindex]
#
#             peaknosieforloop = []
#
#             for i in range(len(intensityforRandomLoop)):
#
#
#                 intensityplusnoise.append(intensityforRandomLoop[i] + random.randint(-5, 5))
#             intensityplusnoise = np.array(intensityplusnoise)
#
#             intensityplusnoise = intensityplusnoise[0:30]
#
#             # traditionalm = traditional_methods(waveLength, intensityplusnoise, y_trainindex, polymerName)
#             ThreeD = TwodataAugmentation(waveLength, intensityplusnoise, y_trainindex, polymerName)
#
#             y_addEmsc = []
#
#             db = "db3"
#             data2D, y_add2D,dataforpoly = ThreeD.generateData(n)
#
#             y_add = []
#             for item in intensityplusnoise:
#                 y_add.append(n)
#
#             x_train = np.concatenate((x_train, intensityplusnoise), axis=0)
#
#             y_train = np.concatenate((y_train, y_add), axis=0)
#
#
#     fig, ax = plt.subplots(nrows=3, ncols=1)
#     font2 = {'family': 'Times New Roman',
#              'weight': 'normal',
#              'size': 30,
#              }
#     for item in dataforpoly:
#         ax[0].plot(waveLength, item, '-r')
#     # for item in data2D:
#     #     ax[1].plot(waveLength, item, '-r')
#     # for item in augmentedSpectrum[0]:
#     #     ax[2].plot(waveLength, item, '-r')
#     labels0 = ax[0].get_xticklabels() + ax[0].get_yticklabels()
#     labels1 = ax[1].get_xticklabels() + ax[1].get_yticklabels()
#     labels2 = ax[2].get_xticklabels() + ax[2].get_yticklabels()
#     ax[0].tick_params(labelsize=15)
#     ax[1].tick_params(labelsize=15)
#     ax[2].tick_params(labelsize=15)
#     # [label0.set_fontname('normal') for label0 in labels0]
#     # [label0.set_fontstyle('normal') for label0 in labels0]
#     ax[0].set_title('EMSC', font2)
#     ax[1].set_title('Original', font2)
#     ax[2].set_title('EMSA', font2)
#     plt.show()