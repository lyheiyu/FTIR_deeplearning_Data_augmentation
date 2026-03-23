import numpy as np
import pandas as pd
from utils import  utils

# !/usr/bin/python

import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from utils import utils
import random
from sklearn.model_selection import train_test_split

def WhittakerSmooth(x, w, lambda_, differences=1):
    '''
    Penalized least squares algorithm for background fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties

    output
        the fitted background vector
    '''
    X = np.matrix(x)
    m = X.size
    i = np.arange(0, m)
    E = eye(m, format='csc')
    D = E[1:] - E[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * D.T * D))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)


def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting

    output
        the fitted background vector
    '''
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
            if (i == itermax): print
            'WARING max iteration reached!'
            break
        w[d >= 0] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    return z
# def normalize(arr: np.ndarray) -> np.ndarray:
#         arr -= arr.min()
#         arr /= arr.max()
#         return arr
class traditional_methods:
    def __init__(self,wavelength,x_train,y_train,polymerName):
        self.wavelength=wavelength
        self.x_train=x_train
        self.y_train=y_train
        self.polyerName=polymerName
    def average(self):
        average=np.zeros(self.x_train.shape[1])
        for item in self.x_train:
            average+=item
        return average/len(self.x_train)

    def generatedata(self,numSynth,n):
        num=numSynth
        data=[]
        y_add=[]
        x_average=self.average()
        for item in self.x_train:

            for j in range(num):

                noise1 = np.random.random(self.wavelength.shape[0]) / 50
                baseline1 = 0.01*random.random()* self.wavelength+0.2   # linear baseline
                y1 = item + baseline1
                #y1=x_average+noise1+baseline1
                data.append(y1)

                y_add.append(n)
        return data,y_add
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import svm
if __name__ == '__main__':
    '''
    Example usage and testing
    '''
    print
    'Testing...'

    from scipy.stats import norm
    import matplotlib.pyplot as plt

    # polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData2('D4_4_publication5.csv', 2, 1763)
    polymerName, waveLength, intensity, polymerID = utils.parseDataForSecondDataset('new_SecondDataset2.csv')
    from FTIR_dataGenerateByConcentratedNet import TwoDimensionalAugmentation

    cmTotal = np.zeros((4, 4))
    m = 0
    t_report = []
    scoreTotal = np.zeros(5)
    for inum in range(20):
    # cmTotal = np.zeros((11, 11))
        x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.7, random_state=inum)

        x = np.arange(0, 1000, 1)
        waveLength = np.array(waveLength, dtype=np.float)
        if waveLength[0] > waveLength[-1]:
            rng = waveLength[0] - waveLength[-1]
        else:
            rng = waveLength[-1] - waveLength[0]
        half_rng = rng / 2
        normalized_wns = (waveLength - np.mean(waveLength)) / half_rng
        #PN = utils.getPN('D4_4_publication5.csv')
        PN = []
        for item in polymerName:
            if item not in PN:
                PN.append(item)
        pID = []
        for item in y_train:
            if item not in pID:
                pID.append(item)
        print(pID)
        if len(pID) <= 3:
            continue
        y_add = []
        data_plot=[]
        for n in range(0,4):

            indicesPS = [i for i, id in enumerate(y_train) if id == n]
            intensityForLoop=x_train[indicesPS]
            idexes = np.arange(len(intensityForLoop))

            indexForIntensity = idexes.take(range(0, 100), mode='wrap')
            intensityRandomLoop=intensityForLoop[indexForIntensity]
            k=[]
            data = []
            for item in intensityRandomLoop:


                #noise1 = np.random.random(waveLength.shape[0]) / 500
                #baseline1 = 0.5*random.randint(-5,5)* normalized_wns+random.randint(-10,10)*0.1  # linear baseline
                #baseline1 = normalized_wns*random.randint(-50,50)*0.1 +  random.randint(-50,50)*0.1 # linear baseline
                #baseline1=baseline1+  random.randint(-50,50)*0.1
                #y1 = item + baseline1
                y1 = item + random.randint(-50,50)*0.1
                data.append(y1)
                data_plot.append(y1)
                y_add.append(n)
                    #print(y1,n)



            #data_plot=np.vstack((data_plot,data))

            x_train = np.concatenate((x_train, data),axis=0)
            print(x_train.shape)
        data_plot=np.array(data_plot)
        from sklearn.preprocessing import normalize
        x_train=normalize(x_train,'max')
        y_train = np.concatenate((y_train, y_add), axis=0)
        print(data_plot.shape)

        KnnClf = KNeighborsClassifier(n_neighbors=2)
        # model=KnnClf.fit(x_train,y_train)
        #model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(244, 258), random_state=1)
        #model.fit(x_train, y_train)
        model = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
        model=model.fit(x_train,y_train)
        y_pre = model.predict(x_test)

        utils.printScore(y_test, y_pre)
        # PN = utils.getPN('D4_4_publication5.csv')


        cm = confusion_matrix(y_test, y_pre)
        from utils import utils
        scores = utils.printScore(y_test, y_pre)

        # SVM_Confusion_matrix = pd.DataFrame(cm)
        cmTotal = cmTotal + cm
        scoreTotal += scores
        m += 1
    # ax[0].plot(x, y1, '-k')
    # modelMLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 128), random_state=1)
    # modelMLP.fit(x_train,y_train)
    # y_pre = modelMLP.predict(x_test)
    #
    # utils.printScore(y_test, y_pre)
    # PN = utils.getPN('D4_4_publication11.csv')
    # t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
    # cm = confusion_matrix(y_test, y_pre)
    # utils.plot_confusion_matrix(cm, PN, 'EMSA_MLP')
    print(m)
    print(scoreTotal / m)
# maxnumber.append(sum(scoreTotal / m) )
#
    cmTotal = cmTotal / m
    utils.plot_confusion_matrix(cmTotal, PN, 'SVM_PLUS')
    # SVM_Confusion_matrix.to_csv('SVM_PLUS.csv')
    #
    # utils.plot_confusion_matrix(cm, PN, 'MLP')

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
    fig, ax = plt.subplots(nrows=2, ncols=1)
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }

    # ax[0].plot(x, y1, '-k')
    for item in data_plot:

        ax[0].plot(waveLength, item, '-r')

    labels0 = ax[0].get_xticklabels() + ax[0].get_yticklabels()

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

    # statisticForMetrics=[]
    # #model = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
    # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(244, 258), random_state=1)
    # models = [model]
    # # cmTotal = np.zeros((12, 12))
    # cmTotal = np.zeros((11, 11))
    # for num,item in enumerate(models):
    #     print(num)
    #     m = 0
    #     t_report=[]
    #     scoreTotal = np.zeros(5)
    #     for i in range(200):
    #         model = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
    #         x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=i)
    #         waveLength = np.array(waveLength, dtype=np.float)
    #         PN = utils.getPN('D4_4_publication5.csv')
    #         y_add = []
    #         data_plot = []
    #         y_e=[]
    #         for it in y_train:
    #             if it not in y_e:
    #                 y_e.append(it)
    #         print(y_e)
    #         if len(y_e)<11 :
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
    #         # from sklearn import svm
    #         #
    #         # KnnClf = KNeighborsClassifier(n_neighbors=2)
    #         # model=KnnClf.fit(x_train,y_train)
    #         #model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(244, 258), random_state=1)
    #         # model.fit(x_train, y_train)
    #
    #         model.fit(x_train, y_train)
    #         y_pre = model.predict(x_test)
    #         y_p=[]
    #         for y_p_e in y_pre:
    #             if y_p_e not in y_p:
    #                 y_p.append(y_p_e)
    #         if len(y_p)<11:
    #             continue
    #         utils.printScore(y_test, y_pre)
    #         PN = utils.getPN('D4_4_publication5.csv')
    #         t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
    #         SVM_report = pd.DataFrame(t)
    #         SVM_report.to_csv('SVM_report5.csv')
    #         cm = confusion_matrix(y_test, y_pre)
    #         #model = tS.spectrumSVM(x_train, y_train, 0.3, 'linear', 'ovo')
    #         y_pre = model.predict(x_test)
    #
    #         cm = confusion_matrix(y_test, y_pre)
    #         cmtemp=np.zeros((12,12))
    #         if len(cm)==11:
    #             continue
    #             # for i in range(len(cm)):
    #             #     cmtemp[i]=np.append(cm[i],[0],axis=0)
    #             #     # cmtemp[i]+=cm[i]
    #             # cmtemp[11,11]=2
    #             # print('tempCM；',cmtemp)
    #             # cmTotal+=cmtemp
    #             # continue
    #         cmTotal=cmTotal+cm
    #         scores=utils.printScore(y_test, y_pre)
    #         m+=1
    #         scoreTotal+=scores
    #         print(m)
    #
    #         t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
    #         utils.mkdir(item.__class__.__name__+'plus_report')
    #         SVM_report = pd.DataFrame(t)
    #
    #         SVM_report.to_csv(item.__class__.__name__+'plus_report/'+item.__class__.__name__ + '_PLUS'+str(i) + '.csv')
    #         #t=np.array(t)
    #         print('Report: -----',t)
    #         t_report+=t
    #     print(scoreTotal/m)
    #     statisticForMetrics.append(scoreTotal/m)
    #
    #     cmTotal=cmTotal/m
    #     print(cmTotal/m)
    #
    #     # utils.plot_confusion_matrix(cmTotal, PN, item.__class__.__name__+'_PCA')
    #     utils.plot_confusion_matrix(cmTotal, PN, 'Baseline argumentation MLP')
    # statisticForMetrics=pd.DataFrame(statisticForMetrics)
    # statisticForMetrics.to_csv('statisticForMetricsPCA.csv')
