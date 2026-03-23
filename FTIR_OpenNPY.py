import numpy as np
import pandas as pd
from sklearn import svm
from FTIR_AugmentationBasedOnEMSA import emsc,EMSA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from PLS import airPLS
from sklearn.preprocessing import normalize,MinMaxScaler,StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import pipeline
from sklearn.model_selection import GridSearchCV
testcorn=np.load('corns/corns_spectra.npy',encoding = "latin1")  #加载文件
# testcorn=pd.DataFrame(testcorn)
testtablets=np.load('tablets/tablets_spectra.npy',encoding = "latin1")  #加载文件
testtablets=pd.DataFrame(testtablets)
testtablets.to_csv('ESMA_tablestpectrum.csv')
corns_markup=np.load('corns/corns_markup.npy',encoding = "latin1")  #加载文件
tablets_markup=np.load('tablets/tablets_markup.npy',encoding = "latin1")
tablets_markup=pd.DataFrame(tablets_markup)
tablets_markup.to_csv('ESMA_tablestMarkup.csv')
labels=corns_markup[:,4]
print(corns_markup)
import random
from FTIR_fit_least_square import generatedataBySperateLSforEach3,\
    generatedataBySperateLSforEach4,generatedataBySperateLSforEach5, generatedataBySperateLSforEach6,generatedataBySperateLSforEach7
# corns_markup=pd.DataFrame(corns_markup)
import matplotlib.pyplot as plt
from utils import utils
testcornBaseline=[]
for item in testcorn:
    item = item - airPLS(item)
    testcornBaseline.append(item)
Mscale=MinMaxScaler()
testcorn=Mscale.fit_transform(testcornBaseline)
# testcorn=np.array(testcornBaseline)
# ss=StandardScaler()
# testcorn=ss.fit_transform(testcornBaseline)
#testcorn=normalize(testcornBaseline, 'max')
wavenumbers=np.load('tablets/wavenumbers.npy',encoding = "latin1")  #加载文件


model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(244, 258), random_state=1)

PN=[]

for item in labels:
    if item not in PN:
        PN.append(item)

waveLength=np.linspace(4000,400,len(testcorn[0]))
cmTotal=np.zeros((len(PN),len(PN)))
m = 0
t_report = []
scoreTotal = np.zeros(5)
for mt in range(20):
    x_train, x_test, y_train, y_test = train_test_split(testcorn, labels, test_size=0.7, random_state=mt)
    #model = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(244, 258), random_state=1)
    datas = []
    datas2 = []
    print(len(y_train))
    if max(y_train)< max(labels) or max(y_test)< max(labels) :
        print(1)
        continue

    for n in range(len(PN)):
        numSynth = 2
        indicesPS = [l for l, id in enumerate(y_train) if id == n]
        intensityForLoop = x_train[indicesPS]
        datas.append(intensityForLoop)
        datas2.append(intensityForLoop)
    # for itr in range(len(PN)):
    #         _, coefs_ = emsc(
    #             datas[itr], waveLength, reference=None,
    #             order=2,
    #             return_coefs=True)
    #
    #         coefs_std = coefs_.std(axis=0)
    #         indicesPS = [l for l, id  in enumerate(y_train) if id == itr]
    #         label = y_train[indicesPS]
    #
    #         reference = datas[itr].mean(axis=0)
    #         emsa = EMSA(coefs_std, waveLength, reference, order=2)
    #
    #         generator = emsa.generator(datas[itr], label,
    #                                    equalize_subsampling=False, shuffle=False,
    #                                    batch_size=300)
    #
    #         augmentedSpectrum = []
    #         for i, batch in enumerate(generator):
    #             if i > 2:
    #                 break
    #             augmented = []
    #             for augmented_spectrum, label in zip(*batch):
    #                 plt.plot(waveLength, augmented_spectrum, label=label)
    #                 augmented.append(augmented_spectrum)
    #             augmentedSpectrum.append(augmented)
    #             # plt.gca().invert_xaxis()
    #             # plt.legend()
    #             # plt.show()
    #         augmentedSpectrum = np.array(augmentedSpectrum)
    #         y_add = []
    #         for item in augmentedSpectrum[0]:
    #             y_add.append(itr)
    #
    #
    #         #augmentedSpectrum[0] = normalize(augmentedSpectrum[0], 'max')
    #         x_train = np.concatenate((x_train, augmentedSpectrum[0]), axis=0)
    #         y_train = np.concatenate((y_train, y_add), axis=0)
    # for n in range(len(PN)):
    #     #for n in nums:
    #
    #     numSynth = 1
    #     indicesPS = [i for i, id in enumerate(y_train) if id == n]
    #     y_trainindex = y_train[indicesPS]
    #     intensityForLoop = x_train[indicesPS]
    #     referenceMean = np.mean(intensityForLoop, axis=0)
    #
    #     intensityplusnoise = []
    #     intensityforRandomLoop = []
    #     idexes = np.arange(len(intensityForLoop))
    #     print(len(idexes))
    #     indexForIntensity = idexes.take(range(0, int(len(idexes)*2)), mode='wrap')
    #     intensityforRandomLoop = intensityForLoop[indexForIntensity]
    #     data2D, label = generatedataBySperateLSforEach6(waveLength, intensityforRandomLoop, n)
    #
    #     peaknosieforloop = []
    #     y_addEmsc = []
    #     db = "db3"
    #     data2D = random.sample(list(data2D), 300)
    #     data2D = np.array(data2D)
    #     dataFromMovingSmooth = data2D
    #
    #     y_add = []
    #     for item in data2D:
    #         y_add.append(n)
    #     dataFromPyw = np.array(data2D)
    #     data2D = np.array(data2D)
    #     ss=StandardScaler()
    #     # data2D=ss.fit_transform(data2D)
    #     x_train = np.concatenate((x_train, data2D), axis=0)
    #     y_train = np.concatenate((y_train, y_add), axis=0)
    m+=1
    model = model.fit(x_train, y_train)
    y_pre = model.predict(x_test)
    plt.clf()
    scores=utils.printScore(y_test, y_pre)
    scoreTotal+=scores
    cm = confusion_matrix(y_test, y_pre)
    cmTotal+=cm
    print(m)
scoreTotal=scoreTotal/m
cmTotal=cmTotal/m
print(scoreTotal)
# Split augmentation corn
utils.plot_confusion_matrix(cmTotal,PN,'Original MLP corn')

#doc = open('1.txt', 'a')  #打开一个存储文件，并依次写入
wavenumbers=pd.DataFrame(corns_markup)
#wavenumbers.to_csv('ESMA_tablets_wavenumbers.csv')
print(wavenumbers)