from FTIR_argumentation_by_traditional_methods import traditional_methods
import pandas as pd
from utils import  utils
import matplotlib.pyplot as plt
# !/usr/bin/python

import pywt
import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from utils import utils
import random
from FTIR_DataConcentratedData3D import randomText
from sklearn.model_selection import train_test_split
from FTIR_DataConcentratedData3D import ThreeDimensionalAugmentation
from FTIR_dataGenerateBasedonLSTM import LSTMAugmentation

from FTIR_dataGenerateByConcentratedNet import  TwoDimensionalAugmentation
from FTIR_AugmentationBasedOnEMSA import emsc
from sklearn.preprocessing import MinMaxScaler
from FTIR_dataAugmentationAVE import VAEDimensionalAugmentation
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from FTIR_PCA import pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re
from FTIR_deriv import cal_2nd_deriv
from FTIR_deriv import cal_3rd_deriv
from FTIR_deriv import cal_deriv
import scipy.signal as sg
from FTIR_AugmentationBasedOnEMSA import EMSA
from FTIR_PolynomialRegression import normalizedWavelength
from FTIR_fit_least_square import generatedataBySperateLSforEach3,\
    generatedataBySperateLSforEach4,generatedataBySperateLSforEach5, generatedataBySperateLSforEach6
if __name__ == '__main__':

    polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData11('D4_4_publication11.csv', 2, 1763)
    # MMScaler = MinMaxScaler()
    # # ss = StandardScaler()
    # intensity=MMScaler.fit_transform(intensity)
    # LDA1=LinearDiscriminantAnalysis(n_components=10)
    # LDA2 = LinearDiscriminantAnalysis(n_components=10)
    # pca = PCA(n_components=0.99)
    # intensity=pca.fit_transform(intensity)
    waveLength=np.array(waveLength)
    print(waveLength)
    if waveLength[0] > waveLength[-1]:
        rng = waveLength[0] - waveLength[-1]
    else:
        rng = waveLength[-1] - waveLength[0]
    half_rng = rng / 2
    normalized_wns = (waveLength - np.mean(waveLength)) / half_rng
    normalWave=normalizedWavelength(waveLength)
    cmTotal=np.zeros((11,11))
    m = 0
    t_report = []
    scoreTotal = np.zeros(5)
    print(cmTotal.shape)
    nums=[0,1,2,7]
    mn=5
    xforplot=np.arange(1,mn+1,1)
    print(xforplot)
    accuracytotal=[]
    nstep=int(len(waveLength)/mn)
    inttotal=[]

    print(intensity.shape)
    for zt in range(mn):
        inteach = []
        for u in range(len(intensity)):
            if zt == m - 1:
                intensitystep = intensity[u][zt * nstep:]
            else:
                intensitystep = intensity[u][zt * nstep:(zt+1)*nstep]
            inteach.append(intensitystep)
        inttotal.append(inteach)
    inteach=np.array(inteach)
    inttotal=np.array(inttotal)
    print(inteach.shape)
    print(inttotal.shape)

    for zt in range(mn):
        for seedNum in range(1):
            # x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.7, random_state=seedNum)
            # x = np.arange(0, 1000, 1)
            #
            # waveLength = np.array(waveLength, dtype=np.float)
            PN = utils.getPN('D4_4_publication11.csv')
            y_add = []
            data_plot = []
            # datas=x_train
            # y_adds=y_train

            x_train, x_test, y_train, y_test = train_test_split(inttotal[zt], polymerID, test_size=0.7,
                                                                random_state=seedNum)
            waveLength = np.array(waveLength, dtype=np.float)
            datas = []
            datas2 = []
            PN = utils.getPN('D4_4_publication11.csv')
            for n in range(len(PN)):
                numSynth = 2
                indicesPS = [l for l, id in enumerate(y_train) if id == n]
                intensityForLoop = x_train[indicesPS]
                datas.append(intensityForLoop)
                datas2.append(intensityForLoop)
            # for itr in range(len(PN)):
            # #for itr in nums:
            #     _, coefs_ = emsc(
            #         datas[itr], waveLength, reference=None,
            #         order=2,
            #         return_coefs=True)
            #
            #     coefs_std = coefs_.std(axis=0)
            #     indicesPS = [l for l, id  in enumerate(y_train) if id == itr]
            #     label = y_train[indicesPS]
            #
            #     reference = datas[itr].mean(axis=0)
            #     emsa = EMSA(coefs_std, waveLength, reference, order=2)
            #
            #     generator = emsa.generator(datas[itr], label,
            #                                equalize_subsampling=False, shuffle=False,
            #                                batch_size=300)
            #
            #     augmentedSpectrum = []
            #     for i, batch in enumerate(generator):
            #         if i > 2:
            #             break
            #         augmented = []
            #         for augmented_spectrum, label in zip(*batch):
            #             plt.plot(waveLength, augmented_spectrum, label=label)
            #             augmented.append(augmented_spectrum)
            #         augmentedSpectrum.append(augmented)
            #         # plt.gca().invert_xaxis()
            #         # plt.legend()
            #         # plt.show()
            #     augmentedSpectrum = np.array(augmentedSpectrum)
            #     y_add = []
            #     for item in augmentedSpectrum[0]:
            #         y_add.append(itr)
            #     from sklearn.preprocessing import normalize
            #
            #     augmentedSpectrum[0] = normalize(augmentedSpectrum[0], 'max')
            #     x_train = np.concatenate((x_train, augmentedSpectrum[0]), axis=0)
            #     y_train = np.concatenate((y_train, y_add), axis=0)

            # for n in range(len(PN)):
            # #for n in nums:
            #
            #     # numSynth = 1
            #     indicesPS = [i for i, id in enumerate(y_train) if id == n]
            #     y_trainindex=y_train[indicesPS]
            #     intensityForLoop = x_train[indicesPS]
            #     referenceMean = np.mean(intensityForLoop, axis=0)
            #
            #     intensityplusnoise=[]
            #     intensityforRandomLoop=[]
            #     idexes=np.arange(len(intensityForLoop))
            #
            #     indexForIntensity=idexes.take(range(0,30),mode='wrap')
            #     intensityforRandomLoop=intensityForLoop[indexForIntensity]
            #     data2D, label = generatedataBySperateLSforEach6(waveLength, intensityforRandomLoop, n)
            #
            #     peaknosieforloop=[]
            #
            #     from sklearn.preprocessing import normalize
            #
            #     y_addEmsc=[]
            #
            #     db = "db3"
            #
            #     data2D = normalize(data2D, 'max')
            #     data2D=random.sample(list(data2D),1500)
            #     data2D=np.array(data2D)
            #     dataFromMovingSmooth = data2D
            #     for i in range(len(dataFromMovingSmooth)):
            #
            #         dataFromMovingSmooth[i] = moving_average(dataFromMovingSmooth[i], 3)
            #     y_add = []
            #     for item in data2D:
            #         y_add.append(n)
            #     dataFromPyw=np.array(data2D)
            #     data2D=np.array(data2D)
            #     # x_train = np.concatenate((x_train, augmentedSpectrum[0]), axis=0)
            #     # y_train = np.concatenate((y_train, y_add2), axis=0)
            #     x_train = np.concatenate((x_train, data2D), axis=0)
            #
            #     y_train = np.concatenate((y_train, y_add), axis=0)

            from sklearn.preprocessing import normalize
            data_plot=np.array(data_plot)
            # x_train=normalize(x_train,'max')

            print(data_plot.shape)
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import classification_report
            from sklearn.metrics import confusion_matrix
            from sklearn.neural_network import MLPClassifier
            from sklearn import svm

            # KnnClf = KNeighborsClassifier(n_neighbors=2)
            # model=KnnClf.fit(x_train,y_train)
            #model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(244, 258), random_state=1)
            # model.fit(x_train, y_train)
            model = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
            print(x_train.shape)
            print(y_train.shape)

            model = model.fit(x_train, y_train)
            y_pre = model.predict(x_test)

            utils.printScore(y_test, y_pre)
            PN = utils.getPN('D4_4_publication11.csv')
            t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
            SVM_report = pd.DataFrame(t)
            SVM_report.to_csv('SVM_report5.csv')
            cm = confusion_matrix(y_test, y_pre)
            from utils import utils

            scores = utils.printScore(y_test, y_pre)

            cmTotal = cmTotal + cm
            scoreTotal += scores
            m+=1
            #utils.plot_confusion_matrix(cm, PN, "Spectral network data_SVM")
            SVM_Confusion_matrix = pd.DataFrame(cm)
            # modelMLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 128), random_state=1)
            # modelMLP.fit(x_train, y_train)
            # y_pre = modelMLP.predict(x_test)
            # utils.printScore(y_test, y_pre)
            # PN = utils.getPN('D4_4_publication11.csv')
            # t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
            # cm = confusion_matrix(y_test, y_pre)
            # #utils.plot_confusion_matrix(cm, PN, 'Spectral network data_MLP')
            #
            # modelKNN = KNeighborsClassifier(n_neighbors=3)
            # modelKNN.fit(x_train, y_train)
            # y_pre = modelKNN.predict(x_test)
            # utils.printScore(y_test, y_pre)
            # PN = utils.getPN('D4_4_publication11.csv')
            # t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
            # cm = confusion_matrix(y_test, y_pre)
            print(m)
        print(scoreTotal / m)
        #maxnumber.append(sum(scoreTotal / m) )
        #
        accuracytotal.append(scoreTotal[2]/m)
        cmTotal = cmTotal / m
        print(cmTotal / m)
        plt.clf()
        utils.plot_confusion_matrix(cmTotal,PN,'Split all dataset agumentation_SVM'+str(zt))
        plt.show()

        # font2 = {'family': 'Times New Roman',
        #          'weight': 'normal',
        #          'size': 30,
        #          }

    plt.plot(xforplot,accuracytotal)
    plt.show()

