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

if __name__ == '__main__':
    polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData2('D4_4_publication5.csv', 2, 1763)
    # MMScaler = MinMaxScaler()
    # # ss = StandardScaler()
    # intensity=MMScaler.fit_transform(intensity)
    # LDA1=LinearDiscriminantAnalysis(n_components=10)
    # LDA2 = LinearDiscriminantAnalysis(n_components=10)
    # pca = PCA(n_components=0.99)
    # intensity=pca.fit_transform(intensity)

    x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.7, random_state=1)
    x = np.arange(0, 1000, 1)

    waveLength = np.array(waveLength, dtype=np.float)
    PN = utils.getPN('D4_4_publication5.csv')
    y_add = []
    data_plot = []
    datas=x_train
    y_adds=y_train
    for n in range(0,11):

        numSynth = 1
        indicesPS = [i for i, id in enumerate(y_train) if id == n]
        y_trainindex=y_train[indicesPS]
        intensityForLoop = x_train[indicesPS]
        # pca = PCA(n_components=0.99)
        # pca.fit(intensityForLoop)
        print(intensityForLoop.shape)
        finalData, reconMat, Vector = pca(intensityForLoop,2)
        print('```````',np.dot(intensityForLoop,Vector[0].T))
        Vector=np.array(Vector)
        for i in range(len(Vector[0])):
            if Vector[0][i]>0.02:
                print("loadings",i,Vector[0][i])
        randomNoise=[]
        for ttt in range(100):
            noise=[]
            for nnn in range(len(Vector[0])):
                if Vector[0][i] > 0.02:
                    noise.append(random.randint(2,4)*0.1)
                if Vector[0][i] <=0.02 :
                    noise.append(0)
            randomNoise.append(noise)
        randomNoise=np.array(randomNoise)
        print(randomNoise.shape)
        randomNoise2 = []
        # for ttt in range(10):
        #     noise2 = []
        #     for nnn in range(len(Vector[1])):
        #         if Vector[1][i] > 0.02:
        #             noise2.append(random.randint(1, 3) * 0.1)
        #         if Vector[1][i] <= 0.02:
        #             noise2.append(0)
        #     randomNoise2.append(noise2)
        # randomNoise2 = np.array(randomNoise2)
        print(randomNoise.shape)
        emscdata, coefs_ = emsc(
            intensityForLoop, waveLength, reference=None,
            order=2,
            return_coefs=True)
        intensityplusnoise=[]
        for i in range(len(randomNoise)):
            rm = random.randint(0,len(intensityForLoop)-1)
            intensityplusnoise.append(intensityForLoop[rm]+randomNoise[i])
        # for i in range(len(randomNoise)):
        #     rm = randomText(len(intensityForLoop), 0, len(intensityForLoop))
        #     intensityplusnoise.append(intensityForLoop[rm]+randomNoise2[i])
        from sklearn.preprocessing import normalize
        intensityplusnoise = normalize(intensityplusnoise, 'max')
        np.random.shuffle(intensityplusnoise)
        # intensityplusnoise=intensityplusnoise[0:20]
        traditionalm = traditional_methods(waveLength, intensityplusnoise, y_trainindex, polymerName)
        ThreeD = ThreeDimensionalAugmentation(waveLength, intensityplusnoise, y_trainindex, polymerName)
        #LSTMD=LSTMAugmentation(waveLength, emscdata, y_trainindex, polymerName)
        TwoD=TwoDimensionalAugmentation(waveLength, intensityplusnoise, y_trainindex, polymerName)
        #VAED = VAEDimensionalAugmentation(waveLength, intensityForLoop, y_trainindex, polymerName)
        #data3,y_add3=ThreeD.generateData(n)
        y_addEmsc=[]
        # for i in range(len(emscdata)):
        #     y_addEmsc.append(n)
        db = "db3"
        data2D,y_add2D=TwoD.generateData(n)
        #data2D, x_test2, y_add2D, y_test2 = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
        # data2D, y_add2D = TwoD.generateData(n)
        w = pywt.Wavelet(db)
        maxlev = pywt.dwt_max_level(len(data2D[0]), w.dec_len)
        print("maximum level is " + str(maxlev))
        # emscdata, coefs_ = emsc(
        #     data2D, waveLength, reference=None,
        #     order=2,
        #     return_coefs=True)
        threshold = 0.01
        dataFromPyw=[]
        # for si in range(len(data2D)):
        #     coeffs = pywt.wavedec(data2D[si], db, level=8)  # 将信号进行小波分解
        #     for ci in range(1, len(coeffs)):
        #         coeffs[ci] = pywt.threshold(coeffs[ci], threshold * max(coeffs[ci]),mode='soft')  # 将噪声滤波
        #
        #     dataFromPyw.append(pywt.waverec(coeffs, db))  # 将信号进行小波重构
        dataFromMovingSmooth = data2D
        for i in range(len(dataFromMovingSmooth)):

            dataFromMovingSmooth[i] = moving_average(dataFromMovingSmooth[i], 3)
        for i in range(len(data2D)):

            data2D[i] = savgol_filter(data2D[i], 43, 2, mode='nearest')
        # for i in range(len(data3)):
        #
        #     data3[i] = savgol_filter(data3[i], 45, 3, mode='nearest')


        #
        # from FTIR_AugmentationBasedOnEMSA import EMSA
        #
        # coefs_std = coefs_.std(axis=0)
        # print(coefs_std)
        # print(coefs_)
        #
        #
        #
        # reference = data2D.mean(axis=0)
        # emsa = EMSA(coefs_std, waveLength, reference, order=0)
        #
        # generator = emsa.generator(data2D, y_add2D,
        #                            equalize_subsampling=False, shuffle=False,
        #                            batch_size=64)
        #
        # augmentedSpectrum = []
        # for i, batch in enumerate(generator):
        #     if i > 2:
        #         break
        #     augmented = []
        #     for augmented_spectrum, label in zip(*batch):
        #         plt.plot(waveLength, augmented_spectrum, label=label)
        #         augmented.append(augmented_spectrum)
        #     augmentedSpectrum.append(augmented)
        #     # plt.gca().invert_xaxis()
        #     # plt.legend()
        #     # plt.show()
        # augmentedSpectrum = np.array(augmentedSpectrum)
        # print(augmentedSpectrum.shape)
        # print(augmentedSpectrum[0].shape)
        # y_add = []
        # for item in augmentedSpectrum[0]:
        #     y_add.append(n)
        # for i in range(len(emscdata)):
        #
        #     emscdata[i] = savgol_filter(emscdata[i], 45, 2, mode='nearest')
        #data2,y_add2=LSTMD.generateData(n)
        # print(data3,y_add3)
        # print(data3.shape, y_add3.shape)
        # data, y_add = traditionalm.generatedata(3,n)
        # data=np.array(data)
        # y_add=np.array(y_add)
        #datas = np.concatenate((datas, emscdata), axis=0)
        dataFromPyw=np.array(data2D)
        # dataFromPyw = dataFromPyw[:, :]
        datas = np.concatenate((datas, data2D), axis=0)

        y_adds = np.concatenate((y_adds, y_add2D), axis=0)
        # datas = np.concatenate((datas, data3), axis=0)
        # y_adds = np.concatenate((y_adds, y_add3), axis=0)
        # datas= np.concatenate((datas, data), axis=0)
        # y_adds= np.concatenate((y_adds, y_add), axis=0)
        # datas= np.concatenate((datas, data2), axis=0)
        # y_adds= np.concatenate((y_adds, y_add2), axis=0)
        # k = []
        # data = []
        # for item in intensityForLoop:
        #     for j in range(numSynth):
        #         # noise1 = np.random.random(waveLength.shape[0]) / 500
        #         baseline1 = 0.005 * random.random() * waveLength + 0.2  # linear baseline
        #         y1 = item + baseline1
        #         data.append(y1)
        #         data_plot.append(y1)
        #         y_add.append(n)
        #         print(y1, n)
        # for item in data:
        #     data_plot.append(item)
        # data_plot=np.vstack((data_plot,data))
    print(y_adds)
    x_train = datas
    y_train = y_adds
    print(x_train.shape)
    # randnum = random.randint(0, 100)
    # random.seed(randnum)
    # random.shuffle(x_train)
    # random.seed(randnum)
    # random.shuffle(y_train)
    x_train, x_test0, y_train, y_test0 = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
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
    print('x_train',x_train.shape)
    model = model.fit(x_train, y_train)
    y_pre = model.predict(x_test)

    utils.printScore(y_test, y_pre)
    PN = utils.getPN('D4_4_publication5.csv')
    t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
    SVM_report = pd.DataFrame(t)
    SVM_report.to_csv('SVM_report5.csv')
    cm = confusion_matrix(y_test, y_pre)
    from utils import utils

    utils.plot_confusion_matrix(cm, PN, 'SVM_PLUS')
    SVM_Confusion_matrix = pd.DataFrame(cm)
    SVM_Confusion_matrix.to_csv('SVM_PLUS.csv')

    utils.plot_confusion_matrix(cm, PN, 'MLP')

    # baseline1 = 5e-4 * waveLength + 0.2  # linear baseline
    # baseline2 = 0.5 * np.sin(np.pi * waveLength / waveLength.max())  # sinusoidal baseline
    # noise1 = np.random.random(waveLength.shape[0]) / 500
    # noise = np.random.random(waveLength.shape[0]) / 500
    # 'Generating simulated experiment'
    # y1 = intensity[0] + baseline1 + noise1
    # y2 = intensity[0] + baseline2 + noise1
    # print
    # from PLS import airPLS
    #
    # y1 = np.array(y1, dtype=np.float)
    # intensityBaseline=[]
    # for item in y1:
    #     item = item - airPLS(item)
    #     intensityBaseline.append(item)
    'Removing baselines'
    # c1 = y1 - airPLS(y1)  # corrected values
    # c2 = y2 - airPLS(y2)  # with baseline removed
    # print
    # 'Plotting results'
    fig, ax = plt.subplots(nrows=4, ncols=1)
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }

    # ax[0].plot(x, y1, '-k')
    for item in dataFromPyw:
        ax[0].plot(waveLength, item, '-r')

    labels0 = ax[0].get_xticklabels() + ax[0].get_yticklabels()
    for item in data2D:
        ax[1].plot(waveLength, item, '-r')
    for item in dataFromMovingSmooth:
        ax[2].plot(waveLength, item, '-r')
    labels0 = ax[1].get_xticklabels() + ax[0].get_yticklabels()
    for item in emscdata:
        ax[3].plot(waveLength, item, '-r')
    # ax[0].tick_params(labelsize=15)
    # [label0.set_fontname('normal') for label0 in labels0]
    # [label0.set_fontstyle('normal') for label0 in labels0]
    ax[0].set_title('Linear baseline', font2)
    # ax[1].plot(waveLength, y2, '-k')
    # ax[1].plot(waveLength, c2, '-r')
    # ax[1].set_title('sinusoidal baseline', font2)
    # plt.tick_params(labelsize=15)
    #
    # labels1 = ax[1].get_xticklabels() + ax[1].get_yticklabels()
    # [label.set_fontname('normal') for label in labels1]
    # [label.set_fontstyle('normal') for label in labels1]
    plt.show()
    print
    'Done!'

    statisticForMetrics = []
    # model = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(244, 258), random_state=1)
    models = [model]
    cmTotal = np.zeros((12, 12))
    # for num, item in enumerate(models):
    #     print(num)
    #     m = 0
    #     t_report = []
    #     scoreTotal = np.zeros(5)
    #     for i in range(200):
    #         model = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
    #         x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=i)
    #         waveLength = np.array(waveLength, dtype=np.float)
    #         PN = utils.getPN('D4_4_publication5.csv')
    #         y_add = []
    #         data_plot = []
    #         y_e = []
    #         for it in y_train:
    #             if it not in y_e:
    #                 y_e.append(it)
    #         print(y_e)
    #         if len(y_e) < 12:
    #             continue
    #         for n in range(len(PN)):
    #             numSynth = 2
    #             indicesPS = [l for l, id in enumerate(y_train) if id == n]
    #             intensityForLoop = x_train[indicesPS]
    #             k = []
    #             data = []
    #             for inten in intensityForLoop:
    #                 for j in range(numSynth):
    #                     # noise1 = np.random.random(waveLength.shape[0]) / 500
    #                     baseline1 = 0.005 * random.random() * waveLength + 0.2  # linear baseline
    #                     y1 = inten + baseline1
    #                     data.append(y1)
    #                     data_plot.append(y1)
    #                     y_add.append(n)
    #
    #             # data_plot=np.vstack((data_plot,data))
    #             x_train = np.concatenate((x_train, data), axis=0)
    #
    #         data_plot = np.array(data_plot)
    #         from sklearn.preprocessing import normalize
    #
    #         # x_train=normalize(x_train,'max')
    #         y_train = np.concatenate((y_train, y_add), axis=0)
    #         # from sklearn.neighbors import KNeighborsClassifier
    #         # from sklearn.metrics import classification_report
    #         # from sklearn.metrics import confusion_matrix
    #         from sklearn.neural_network import MLPClassifier
    #
    #         # from sklearn import svm
    #         #
    #         # KnnClf = KNeighborsClassifier(n_neighbors=2)
    #         # model=KnnClf.fit(x_train,y_train)
    #         # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(244, 258), random_state=1)
    #         # model.fit(x_train, y_train)
    #
    #         model.fit(x_train, y_train)
    #         y_pre = model.predict(x_test)
    #         y_p = []
    #         for y_p_e in y_pre:
    #             if y_p_e not in y_p:
    #                 y_p.append(y_p_e)
    #         if len(y_p) < 12:
    #             continue
    #         utils.printScore(y_test, y_pre)
    #         PN = utils.getPN('D4_4_publication5.csv')
    #         t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
    #         SVM_report = pd.DataFrame(t)
    #         SVM_report.to_csv('SVM_report5.csv')
    #         cm = confusion_matrix(y_test, y_pre)
    #         # model = tS.spectrumSVM(x_train, y_train, 0.3, 'linear', 'ovo')
    #         y_pre = model.predict(x_test)
    #
    #         cm = confusion_matrix(y_test, y_pre)
    #         cmtemp = np.zeros((12, 12))
    #         if len(cm) == 11:
    #             continue
    #             # for i in range(len(cm)):
    #             #     cmtemp[i]=np.append(cm[i],[0],axis=0)
    #             #     # cmtemp[i]+=cm[i]
    #             # cmtemp[11,11]=2
    #             # print('tempCM；',cmtemp)
    #             # cmTotal+=cmtemp
    #             # continue
    #         cmTotal = cmTotal + cm
    #         scores = utils.printScore(y_test, y_pre)
    #         m += 1
    #         scoreTotal += scores
    #         print(m)
    #
    #         t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
    #         utils.mkdir(item.__class__.__name__ + 'plus_report')
    #         SVM_report = pd.DataFrame(t)
    #
    #         SVM_report.to_csv(
    #             item.__class__.__name__ + 'plus_report/' + item.__class__.__name__ + '_PLUS' + str(i) + '.csv')
    #         # t=np.array(t)
    #         print('Report: -----', t)
    #         t_report += t
    #     print(scoreTotal / m)
    #     statisticForMetrics.append(scoreTotal / m)
    #
    #     cmTotal = cmTotal / m
    #     print(cmTotal / m)
    #
    #     # utils.plot_confusion_matrix(cmTotal, PN, item.__class__.__name__+'_PCA')
    #     utils.plot_confusion_matrix(cmTotal, PN, 'Baseline argumentation MLP')
    # statisticForMetrics = pd.DataFrame(statisticForMetrics)
    # statisticForMetrics.to_csv('statisticForMetricsPCA.csv')
