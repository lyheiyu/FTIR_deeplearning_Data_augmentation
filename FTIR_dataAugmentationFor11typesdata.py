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

if __name__ == '__main__':

    polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData11('D4_4_publication11.csv', 2, 1763)
    # MMScaler = MinMaxScaler()
    # # ss = StandardScaler()
    # intensity=MMScaler.fit_transform(intensity)
    # LDA1=LinearDiscriminantAnalysis(n_components=10)
    # LDA2 = LinearDiscriminantAnalysis(n_components=10)
    # pca = PCA(n_components=0.99)
    # intensity=pca.fit_transform(intensity)

    cmTotal=np.zeros((11,11))
    m = 0
    t_report = []
    scoreTotal = np.zeros(5)
    print(cmTotal.shape)
    for seedNum in range(20):
        x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.7, random_state=seedNum)
        x = np.arange(0, 1000, 1)

        waveLength = np.array(waveLength, dtype=np.float)
        PN = utils.getPN('D4_4_publication11.csv')
        y_add = []
        data_plot = []
        datas=x_train
        y_adds=y_train

        for n in range(0,11):

            numSynth = 1
            indicesPS = [i for i, id in enumerate(y_train) if id == n]
            y_trainindex=y_train[indicesPS]
            intensityForLoop = x_train[indicesPS]
            referenceMean = np.mean(intensityForLoop, axis=0)
            # peaks=[]
            # for i in range(len(intensityForLoop)):
            #     peaks.append(sg.argrelmax(intensityForLoop[i])[0])
            # troughs=[]
            # for i in range(len(intensityForLoop)):
            #     troughs.append(sg.argrelmin(intensityForLoop[i])[0])
            # noiseforpeaks=[]
            # for mt in range(len(peaks)):
            #     noise = []
            #     for ittt in range(len(intensityForLoop[0])):
            #             for item in range(len(peaks[mt])):
            #                 if ittt==item:
            #                     noise.append(random.randint(1,2)*0.1)
            #                 else:
            #                     noise.append(0)
            #     noiseforpeaks.append(noise)
            # for mt in range(len(troughs)):
            #     for irrr in range(len(intensityForLoop[0])):
            #             for item in range(len(troughs[mt])):
            #                 if irrr==item:
            #                     noiseforpeaks[mt][irrr]=-random.randint(1,2)*0.1


            # pca = PCA(n_components=0.99)
            # pca.fit(intensityForLoop)
            deirves=[]
            deirves2 = []
            deirves3=[]
            for item in intensityForLoop:
                deiv = cal_deriv(waveLength, item)
                deiv2=cal_2nd_deriv(waveLength,item)
                deiv3=cal_3rd_deriv(waveLength,item)
                deirves3.append(deiv3)
                deirves2.append(deiv2)
                deirves.append(deiv)

            deirves=np.array(deirves)
            print(deirves.shape)
            deirves2 = np.array(deirves2)
            print(deirves.shape)
            deirves3 = np.array(deirves3)
            print(deirves.shape)
            # finalData, reconMat, Vector = pca(intensityForLoop,3)
            # print('```````',np.dot(intensityForLoop,Vector[0].T))
            # Vector=np.array(Vector)
            # for i in range(len(Vector[0])):
            #     if Vector[0][i]>0.02:
            #         print("loadings",i,Vector[0][i])
            # randomNoise=[]
            # for ttt in range(30):
            #     noise=[]
            #     for nnn in range(len(Vector[0])):
            #         if Vector[0][i] > 0:
            #             noise.append(random.randint(2,3)*0.3)
            #
            #         if Vector[0][i] <=0 :
            #             noise.append(random.randint(2,3)*(-0.3))
            #     randomNoise.append(noise)
            # randomNoise=np.array(randomNoise)
            #randomNoise2 = []
            # for ttt in range(50):
            #     noise2 = []
            #     for nnn in range(len(Vector[1])):
            #         if Vector[1][i] > 0.02:
            #             noise2.append(random.randint(2, 3) * 0.1)
            #         if Vector[1][i] <= 0.02:
            #             noise2.append(0)
            #     randomNoise2.append(noise2)
            # randomNoise2 = np.array(randomNoise2)
            emscdata, coefs_ = emsc(
                intensityForLoop, waveLength, reference=None,
                order=2,
                return_coefs=True)
            intensityplusnoise=[]
            intensityforRandomLoop=[]
            idexes=np.arange(len(intensityForLoop))

            indexForIntensity=idexes.take(range(0,200),mode='wrap')
            intensityforRandomLoop=intensityForLoop[indexForIntensity]
            deirvesforloopindex = idexes.take(range(0, 200), mode='wrap')
            deirvesforloop = deirves[deirvesforloopindex]
            deirves2forloopindex = idexes.take(range(0, 200),mode='wrap')
            print(deirves2)
            print(deirves2forloopindex)
            deirves2forloop2=deirves2[deirves2forloopindex]
            deirvesforloopindex = idexes.take(range(0, 200), mode='wrap')
            deirvesforloop3 = deirves3[deirvesforloopindex]


            peaknosieforloop=[]
            # for pre in range(5):
            #     for pr in range(len(noiseforpeaks)):
            #         peaknosieforloop.append(noiseforpeaks[pr])
            randomDeivs=[]
            for mmm in range(5):
                for mtr in range(len(deirves)):
                    randomDeivs.append(deirves[mtr])
            randomDeivs=randomDeivs[0:10]

            for i in range(len(intensityforRandomLoop)):
                noiseforeahc=[]
            #     for eachj in range(len(deirvesforloop[i])):
            #         if eachj>=len(deirvesforloop[i])-2:
            #             noiseforeahc.append(intensityforRandomLoop[i][eachj] + random.randint(9,11)*deirves2forloop2[i][eachj])
            #         else:
            #             noiseforeahc.append(intensityforRandomLoop[i][eachj] + random.randint(9,11)*deirves2forloop2[i][eachj])
            #     intensityplusnoise.append(np.array(noiseforeahc) + random.randint(-4, 4) * 0.3)
                for eachj in range(len(deirvesforloop[i])):
                    if eachj<1760:
                        # if eachj <=len(deirvesforloop[i])/5:
                        #     if deirves2forloop2[i][eachj]<0 and deirves2forloop2[i][eachj+1]>0:
                        #         intensityforRandomLoop[i][eachj]+=np.random.randint(2,3)
                        # if len(deirvesforloop[i])/5<eachj <=2*len(deirvesforloop[i])/5:
                        #     if deirves2forloop2[i][eachj]<0 and deirves2forloop2[i][eachj+1]>0:
                        #         intensityforRandomLoop[i][eachj]+=np.random.randint(5,6)
                        # if 2*len(deirvesforloop[i])/5<eachj <=3*len(deirvesforloop[i])/5:
                        #     if deirves2forloop2[i][eachj]<0 and deirves2forloop2[i][eachj+1]>0:
                        #         intensityforRandomLoop[i][eachj]+=np.random.randint(4,6)
                        # if 3*len(deirvesforloop[i])/5<eachj <=4*len(deirvesforloop[i])/5:
                        #     if deirves2forloop2[i][eachj]<0 and deirves2forloop2[i][eachj+1]>0:
                        intensityforRandomLoop[i][eachj]+=np.random.randint(7,8)*0.5
                    else:

                        intensityforRandomLoop[i][eachj]+=np.random.randint(7,8)*0.3


                intensityplusnoise.append(intensityforRandomLoop[i]+ random.randint(-2, 2) * 0.3)

            # for i in range(len(intensityforRandomLoop)):
            #     noiseforeahc = []
            #     for eachj in range(len(deirvesforloop[i])):
            #
            #         noiseforeahc.append(referenceMean[eachj] + peaknosieforloop[i][eachj])
            #
            #
            #     intensityplusnoise.append(np.array(noiseforeahc) + random.randint(-1, 1) * 0.3)
                    # intensityplusnoise.append(referenceMean+randomDeivs[i]+random.randint(3,4)*0.1)
            # for i in range(len(randomNoise2)):
            #     rm = random.randint(0,len(intensityForLoop)-1)
            #     intensityplusnoise.append(intensityForLoop[rm]+randomNoise2[i])
            from sklearn.preprocessing import normalize
            intensityplusnoise = normalize(intensityplusnoise, 'max')
            #np.random.shuffle(intensityplusnoise)
            intensityplusnoise=intensityplusnoise[0:20]
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
            # data2D,y_add2D=TwoD.generateData(n)
            data2D=intensityplusnoise
            y_add2D=[]
            for i in range(len(intensityplusnoise)):
                y_add2D.append(n)
            #data2D, x_test2, y_add2D, y_test2 = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
            # data2D, y_add2D = TwoD.generateData(n)
            # w = pywt.Wavelet(db)
            # maxlev = pywt.dwt_max_level(len(data2D[0]), w.dec_len)
            # print("maximum level is " + str(maxlev))
            # emscdata, coefs_ = emsc(
            #     data2D, waveLength, reference=None,
            #     order=2,
            #     return_coefs=True)
            # threshold = 0.01
            # dataFromPyw=[]
            # for si in range(len(data2D)):
            #     coeffs = pywt.wavedec(data2D[si], db, level=8)  # 将信号进行小波分解
            #     for ci in range(1, len(coeffs)):
            #         coeffs[ci] = pywt.threshold(coeffs[ci], threshold * max(coeffs[ci]),mode='soft')  # 将噪声滤波
            #
            #     dataFromPyw.append(pywt.waverec(coeffs, db))  # 将信号进行小波重构
            #data2D = normalize(data2D, 'max')
            dataFromMovingSmooth = data2D
            for i in range(len(dataFromMovingSmooth)):

                dataFromMovingSmooth[i] = savgol_filter(dataFromMovingSmooth[i], 43, 2, mode='nearest')
            # for i in range(len(dataFromMovingSmooth)):
            #
            #     dataFromMovingSmooth[i] = moving_average(dataFromMovingSmooth[i], 3)
            data2D=dataFromMovingSmooth
            # for i in range(len(data2D)):
            #
            #     data2D[i] = savgol_filter(data2D[i], 43, 2, mode='nearest')
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
        x_train = datas
        y_train = y_adds

    # randnum = random.randint(0, 100)
    # random.seed(randnum)
    # random.shuffle(x_train)
    # random.seed(randnum)
    # random.shuffle(y_train)
    #x_train, x_test0, y_train, y_test0 = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
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
    cmTotal = cmTotal / m
    print(cmTotal / m)
    utils.plot_confusion_matrix(cmTotal,PN,'Spectral network data_SVM')


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
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 128), random_state=1)
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
