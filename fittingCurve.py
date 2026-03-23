import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
from sklearn.neighbors import  KNeighborsClassifier
import math
from pylab import mpl
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.signal import savgol_filter
from PLS import airPLS
from sklearn.preprocessing import MinMaxScaler
def readData(fileName):
    dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False)
    polymerName = dataset.iloc[1:, 1]
    polymerId= dataset.iloc[1:,2]
    intensity = dataset.iloc[1:271, 3:1179]
    waveLength=dataset.iloc[0,3:1179]
    polymerName=np.array(polymerName)
    polymerId=np.array(polymerId)
    intensity=np.array(intensity)
    waveLength=np.array(waveLength)
    # for item in waveLength:
    #     item =str(item)
    #     print(item)
    return polymerName,waveLength,intensity,polymerId

def parseData3(fileName, begin, end):
    #dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False)
    dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False)
    polymerName=dataset.iloc[1:271,1]
    polymerID=dataset.iloc[1:271,2]
    # waveLength=dataset.iloc[0,begin:end]
    intensity=dataset.iloc[1:271,begin:end]

    # polymerName= np.array(polymerName)
    polymerID =np.array(polymerID)
    # waveLength=np.array(waveLength)
    # intensity =np.array(intensity)
    # print(dataset)
    #polymerName = dataset.iloc[1:971, 1]
    # print(polymerName)
    waveLength = dataset.iloc[0, begin:end]



    #intensity = dataset.iloc[1:971, begin:end]
    intensity = np.array(intensity)
    for j in range(len(intensity)):

        for i in range(len(intensity[j])):
            if intensity[j][i] == '':
                intensity[j][i] = 0.000
    for j in range(len(intensity)):

        for i in range(len(intensity[j])):
            intensity[j][i] = np.float(intensity[j][i])

            #print(it.__class__)
        #print(item)
    polymerName = np.array(polymerName)
    waveLength = np.array(waveLength)
    for i in range(len(waveLength)):
        waveLength[i]=float(waveLength[i])
    return polymerName, waveLength, intensity,polymerID
def parseData2(fileName, begin, end):
    dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False)
    polymerName = dataset.iloc[1:971, 1]
    polymerIDfromDataset = dataset.iloc[1:971, 1764]
    # print(polymerIDfromDataset)
    waveLength = dataset.iloc[0, begin:end]

    intensity = dataset.iloc[1:971, begin:end]
    polymerName = np.array(polymerName)
    waveLength = np.array(waveLength)
    # polymerNameList = {}.fromkeys(polymerName).keys()
    # polymerNameList = []
    # pList = list(set(polymerName))
    # for item in pList:
    #     polymerNameList.append(item)
    # polymerNameList = np.array(polymerNameList)
    # polymerNameID = []
    # for i in range(len(polymerNameList)):
    #     polymerNameID.append(i + 1)
    # polymerNameData = []
    # for item in polymerName:
    #     for i in range(len(pList)):
    #         if item == pList[i]:
    #             polymerNameData.append(i + 1)

    intensity = np.array(intensity)
    for item in intensity:

        for i in range(len(item)):
            if item[i] == '':
                item[i] = 0
    polymerIDfromDataset=np.array(polymerIDfromDataset)
    return polymerName, waveLength, intensity, polymerIDfromDataset
def fitCurve(waveLength,intensity):
    #x = np.linspace(4000, int(waveLength.max()), num=int(((4000 - int(waveLength.max())) / waveLength.shape[0]) * 1176))
    x = np.linspace(4000, int(waveLength.max()), num=int(((4000 - int(waveLength.max())) /
                                                          (int(waveLength.max()-waveLength.min()))) * waveLength.shape[0]))

    x2 = np.linspace(int(waveLength.min()), 500, num=int(((int(waveLength.min()-500)) /
                                                          (int(waveLength.max()-waveLength.min()))) * waveLength.shape[0]))
    y2 = np.zeros((len(intensity), x2.shape[0]))
    #print(x.shape, y.shape, intensity.shape)
    #x = np.append(x, waveLength3)
    x = np.append(x, x2)
    return x

if __name__ == '__main__':
    from utils import utils
    #polymerName, waveLength, intensity, polymerID,x_each,y_each = utils. parseData2('D4_4_publication5.csv', 2, 1763)
    #polymerName2, waveLength2, intensity2, polymerID2 = readData('216_2018_1156_MOESM5_ESM.csv')
    #polymerName3, waveLength3, intensity3, polymerID3 = parseData3('216_2018_1156_MOESM5_ESM.csv',3,1179)
    polymerName, waveLength, intensity, polymerID,x_each,y_each =utils.parseData2('D4_4_publication5.csv', 2, 1763)
    for i in range(len(intensity)):
        intensity[i, :] = savgol_filter(intensity[i, :], 33, 2, mode='nearest')
   # print(intensity3.shape,waveLength3.shape)
   #  for item in intensity3[1]:
   #      print(item.__class__)
   #  for item in intensity[1]:
   #      print(item.__class__)
    #plt.plot(waveLength3,intensity3[1])
    #print(intensity3.shape)

    polymerNameSet = []
    # polymerIdSet=[]
    minMax = MinMaxScaler()
    polymerNameSet2 = []
    # polymerIdSet=[]
    for polymerN in polymerName:
        if polymerN not in polymerNameSet2:
            polymerNameSet2.append(polymerN)
    polymerNameSet2 = np.array(polymerNameSet2)
    print(polymerNameSet2)
    print("12123",polymerNameSet)
    colors = ['c', 'b', 'g', 'r', 'm', 'y', '#377eb8', 'darkviolet', 'olive',
              'tan', 'lightgreen', 'gold', 'cyan', 'magenta', 'pink',
              'crimson', 'navy', 'cadetblue','#ffffff','#bbbbbb','#aaaaaa','#dddddd','#123123','#321321',
              '#533ef1','#abeabe','c', 'b', 'g', 'r', 'm', 'y']
    #plt.ylim(0, 1)
    #plt.xlim(4000, 500)
    fig, ax = plt.subplots()
    plt.tick_params(labelsize=20)
    plt.xlim(4000, 500)

    latent_dim = 100
    numSynth = 15
    generatorPS = load_model("Poly(styrene) Generator")
    random_latent_vectors = tf.random.normal(shape=(numSynth, latent_dim))
    generated_specs_ps = generatorPS(random_latent_vectors)
    print(generated_specs_ps.shape)
    generated_specs_ps = generated_specs_ps.numpy()
    data = generated_specs_ps.reshape((numSynth, 1761))
    # intensity = minMax.fit_transform(intensity)
    # data=minMax.fit_transform(data)
    data = normalize(data, 'max')
    dataBaseline=[]
    for item in data:
        item = item - airPLS(item)
        dataBaseline.append(item)
    data = normalize(data, 'max')

    # def normalize(arr: np.ndarray) -> np.ndarray:
    #     arr -= arr.min()
    #     arr /= arr.max()
    #     return arr
    #


    # for i in range(numSynth):
    #
    #     data[i, :] = savgol_filter(data[i, :],33, 2, mode='nearest')
    #intensity = normalize(intensity, 'l2')
    latent_dim = 100
    numSynth = 20
    generatorPF = load_model("selectedPoly(ethylene) Generateor 50000")
    random_latent_vectors = tf.random.normal(shape=(numSynth, latent_dim))
    generated_specs_PF = generatorPF(random_latent_vectors)
    generated_specs_PF = generated_specs_PF.numpy()
    data2 = generated_specs_PF.reshape((numSynth, 1761))
    dataBaseline2 = []
    for item in data2:
        item = item - airPLS(item)
        dataBaseline2.append(item)

    data2 = normalize(dataBaseline2, 'max')
    # for i in range(numSynth):
    #     data2[i, :] = savgol_filter(data2[i, :], 33, 2, mode='nearest')
    # for j in range(len(polymerName)):
    #     #for i in range(len(polymerNameSet2)):
    #     if polymerName[j] == 'Poly(ethylene) like':
    #         ax.plot(waveLength, intensity[j], color=colors[0])
    # for i in range(len(data)):
    #     #for i in range(len(polymerNameSet2)):
    #
    #     ax.plot(waveLength, data[i], color=colors[1])
    for i in range(len(data2)):
        #for i in range(len(polymerNameSet2)):

        ax.plot(waveLength, data2[i], color='b')
        print(i)
    # for j in range(len(polymerName)):
    #
    #     if polymerName[j] == 'Poly(ethylene)':
    #         # print(polymerID[j])
    #         ax.plot(waveLength, intensity[j], color=colors[4])
    #     if polymerName[j] == 'Poly(ethylene) + fouling':
    #         # print(polymerID[j])
    #         ax.plot(waveLength, intensity[j], color=colors[5])
    plt.legend()
    plt.xlabel('Wavelength',size=30)
    plt.ylabel('Intensity',size=30)
    plt.title('PE',size=30)
    #plt.legend()
    plt.show()
    normalizer = normalize(norm="max")

    intensity=normalizer.transform(intensity)
    model=KNeighborsClassifier(n_neighbors=5)
    model.fit(intensity,polymerName)

    #plt.ylim(intensity.min(), intensity.max())
    fig,ax=plt.subplots()


    #ax.tick_params(labelsize=30)
    plt.xlim(4000,500)
    plt.ylim(0,110)

    x_pred = np.linspace(4000, 500, num=waveLength.shape[0])

    print(intensity[0].shape)
    # print(f1)
    # x_pred = np.linspace(3000, 3200, num=1000)


