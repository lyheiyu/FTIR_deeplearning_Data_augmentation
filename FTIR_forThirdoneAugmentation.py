
import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
from math import isnan
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import pandas as pd
from typing import Union as U, Tuple as T
from utils import utils
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from scipy.signal import savgol_filter
from FTIR_dataGenerateByPolyfit import TwodataAugmentation
from FTIR_PolynomialRegression import LSTMPolynomiaRegression
def emsc(spectra: np.ndarray, wavenumbers: np.ndarray, order: int = 2,
         reference: np.ndarray = None,
         constituents: np.ndarray = None,
         return_coefs: bool = False) -> U[np.ndarray, T[np.ndarray, np.ndarray]]:
    """
    Preprocess all spectra with EMSC
    :param spectra: ndarray of shape [n_samples, n_channels]
    :param wavenumbers: ndarray of shape [n_channels]
    :param order: order of polynomial
    :param reference: reference spectrum
    :param constituents: ndarray of shape [n_consituents, n_channels]
    Except constituents it can also take orthogonal vectors,
    for example from PCA.
    :param return_coefs: if True returns coefficients
    [n_samples, n_coeffs], where n_coeffs = 1 + len(costituents) + (order + 1).
    Order of returned coefficients:
    1) b*reference +                                    # reference coeff
    k) c_0*constituent[0] + ... + c_k*constituent[k] +  # constituents coeffs
    a_0 + a_1*w + a_2*w^2 + ...                         # polynomial coeffs
    :return: preprocessed spectra
    """
    if reference is None:
        reference = np.mean(spectra, axis=0)
    print(spectra)
    print(reference)
    reference = reference[:, np.newaxis]

    # squeeze wavenumbers to approx. range [-1; 1]
    # use if else to support uint types
    if wavenumbers[0] > wavenumbers[-1]:
        rng = wavenumbers[0] - wavenumbers[-1]
    else:
        rng = wavenumbers[-1] - wavenumbers[0]
    half_rng = rng / 2
    normalized_wns = (wavenumbers - np.mean(wavenumbers)) / half_rng

    polynomial_columns = [np.ones(len(wavenumbers))]
    for j in range(1, order + 1):
        polynomial_columns.append(normalized_wns ** j)
    polynomial_columns = np.stack(polynomial_columns).T

    # spectrum = X*coefs + residues
    # least squares -> A = (X.T*X)^-1 * X.T; coefs = A * spectrum
    if constituents is None:
        columns = (reference, polynomial_columns)
    else:
        columns = (reference, constituents.T, polynomial_columns)

    X = np.concatenate(columns, axis=1)
    A = np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T)

    spectra_columns = spectra.T
    coefs = np.dot(A, spectra_columns)
    residues = spectra_columns - np.dot(X, coefs)

    preprocessed_spectra = (reference + residues/coefs[0]).T

    if return_coefs:
        return preprocessed_spectra, coefs.T

    return preprocessed_spectra
def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re
class EMSA:
    """
    Extended Multiplicative Signal Augmentation
    Generates balanced batches of augmentated spectra
    """

    def __init__(self, std_of_params, wavenumbers, reference, order=2):
        """
        :param std_of_params: array of length (order+2), which
        :param reference: reference spectrum that was used in EMSC model
        :param order: order of emsc
        contains the std for each coefficient
        """
        self.order = order
        self.std_of_params = std_of_params
        self.ref = reference
        self.X = None
        self.A = None
        self.__create_x_and_a(wavenumbers)

    def generator(self, spectra, labels,
                  equalize_subsampling=False, shuffle=True,
                  batch_size=32):
        """ generates batches of transformed spectra"""
        spectra = np.asarray(spectra)
        labels = np.asarray(labels)

        if self.std_of_params is None:
            coefs = np.dot(self.A, spectra.T)
            self.std_of_params = coefs.std(axis=1)

        if equalize_subsampling:
            indexes = self.__rearrange_spectra(labels)
        else:
            indexes = np.arange(len(spectra))

        cur = 0
        while True:
            if shuffle:
                si = indexes[np.random.randint(len(indexes),
                                               size=batch_size)]
            else:
                si = indexes.take(range(cur, cur + batch_size),
                                  mode='wrap')
                cur += batch_size

            yield self.__batch_transform(spectra[si]), labels[si]

    def __rearrange_spectra(self, labels):
        """ returns indexes of data rearranged in the way of 'balance'"""
        classes = np.unique(labels, axis=0)

        if len(labels.shape) == 2:
            grouped = [np.where(np.all(labels == l, axis=1))[0]
                       for l in classes]
        else:
            grouped = [np.where(labels == l)[0] for l in classes]
        iters_cnt = max([len(g) for g in grouped])

        indexes = []
        for i in range(iters_cnt):
            for g in grouped:
                # take cyclic sample from group
                indexes.append(np.take(g, i, mode='wrap'))

        return np.array(indexes)

    def __create_x_and_a(self, wavenumbers):
        """
        Builds X matrix from spectra in such way that columns go as
        reference w^0 w^1 w^2 ... w^n, what corresponds to coefficients
        b, a, d, e, ...
        and caches the solution self.A = (X^T*X)^(-1)*X^T
        :param spectra:
        :param wavenumbers:
        :return: nothing, but creates two self.X and self.A
        """
        # squeeze wavenumbers to approx. range [-1; 1]
        # use if else to support uint types
        if wavenumbers[0] > wavenumbers[-1]:
            rng = wavenumbers[0] - wavenumbers[-1]
        else:
            rng = wavenumbers[-1] - wavenumbers[0]
        half_rng = rng / 2
        normalized_wns = (wavenumbers - np.mean(wavenumbers)) / half_rng

        self.polynomial_columns = [np.ones_like(wavenumbers)]
        for j in range(1, self.order + 1):
            self.polynomial_columns.append(normalized_wns ** j)

        self.X = np.stack((self.ref, *self.polynomial_columns), axis=1)
        self.A = np.dot(np.linalg.pinv(np.dot(self.X.T, self.X)), self.X.T)

    def __batch_transform(self, spectra):
        spectra_columns = spectra.T

        # b, a, d, e, ...

        coefs = np.dot(self.A, spectra_columns)
        residues = spectra_columns - np.dot(self.X, coefs)

        new_coefs = coefs.copy()

        # wiggle coefficients
        for i in range(len(coefs)):
            new_coefs[i] += np.random.normal(0,
                                             self.std_of_params[i],
                                             len(spectra))

        # Fix if multiplication parameter sampled negative
        mask = new_coefs[0] <= 0
        if np.any(mask):
            # resample multiplication parameter to be positive
            n_resamples = mask.sum()
            new_coefs[0][mask] = np.random.uniform(0, coefs[0][mask],
                                                   n_resamples)


        return (np.dot(self.X, new_coefs) + residues * new_coefs[0] / coefs[0]).T
class TestKNN:
    def __init__(self):
        pass
    def parseData2(fileName,begin,end):

        dataset = pd.read_csv(fileName, header=None, encoding='latin-1',keep_default_na=False)
        # polymerName=dataset.iloc[1:271,1]
        # polymerID=dataset.iloc[1:271,2]
        # waveLength=dataset.iloc[0,begin:end]
        # intensity=dataset.iloc[1:271,begin:end]
        # polymerName= np.array(polymerName)
        # polymerID =np.array(polymerID)
        # waveLength=np.array(waveLength)
        # intensity =np.array(intensity)
        # print(dataset)
        polymerName=dataset.iloc[1:971,1]
        #print(polymerName)
        waveLength=dataset.iloc[0,begin:end]
        print(waveLength)
        intensity=dataset.iloc[1:971,begin:end]
        polymerName= np.array(polymerName)

        waveLength=np.array(waveLength)

        intensity =np.array(intensity)
        return polymerName,waveLength,intensity
    def spectrumKNN(x,y,number):
        model= KNeighborsClassifier(n_neighbors=number)

        model.fit(x,y)

        return model
if __name__ == '__main__':

    tk=TestKNN
    #this is for the 216_2018.csv
    polymerName, waveLength, intensity=tk.parseData2('216_2018_1156_MOESM5_ESM2.csv',3,1179)
    PN=[]
    for item in polymerName:
        if item not in PN:
            PN.append(item)
    polymerID = []

    for i in range(len(PN)):
        for item in polymerName:
            if item==PN[i]:
                polymerID.append(i)
    print(polymerID)

    print('PN adsfadsfasdf',PN,len(PN))


    for item in intensity:

        for i in range(len(item)):
            if item[i]=='':

                item[i]=np.float(0)


                #print(waveIntensity)
                #print(item)
    intensity=np.array(intensity,dtype=np.float32)
    polymerID=np.array(polymerID)
    for seedNum in range(1):
        x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.7, random_state=seedNum)
        xtraindata = x_train
        ytraindata = y_train
        waveLength = np.array(waveLength, dtype=np.float)
        datas = []
        datas2 = []
        PN = []
        for item in polymerName:
            if item not in PN:
                PN.append(item)
        pID = []
        for item in y_train:
            if item not in pID:
                pID.append(item)
        print(pID)
        if len(pID) <= 15:
            continue
        for n in range(len(PN)):
            numSynth = 2
            indicesPS = [l for l, id in enumerate(y_train) if id == n]
            intensityForLoop = x_train[indicesPS]
            datas.append(intensityForLoop)
            datas2.append(intensityForLoop)
        for itr in range(15):
            _, coefs_ = emsc(
                datas[itr], waveLength, reference=None,
                order=2,
                return_coefs=True)

            coefs_std = coefs_.std(axis=0)
            indicesPS = [l for l, id in enumerate(y_train) if id == itr]
            label = y_train[indicesPS]

            reference = datas[itr].mean(axis=0)
            emsa = EMSA(coefs_std, waveLength, reference, order=2)

            generator = emsa.generator(datas[itr], label,
                                       equalize_subsampling=False, shuffle=False,
                                       batch_size=100)

            augmentedSpectrum = []
            for i, batch in enumerate(generator):
                if i > 2:
                    break
                augmented = []
                for augmented_spectrum, label in zip(*batch):
                    plt.plot(waveLength, augmented_spectrum, label=label)
                    augmented.append(augmented_spectrum)
                augmentedSpectrum.append(augmented)
                # plt.gca().invert_xaxis()
                # plt.legend()
                # plt.show()
            augmentedSpectrum = np.array(augmentedSpectrum)
            y_add = []
            for item in augmentedSpectrum[0]:
                y_add.append(itr)
            from sklearn.preprocessing import normalize

            augmentedSpectrum[0] = normalize(augmentedSpectrum[0], 'max')
            x_train = np.concatenate((x_train, augmentedSpectrum[0]), axis=0)
            y_train = np.concatenate((y_train, y_add), axis=0)
    accuracies=[]
    index=[]
    for i in range(3,10):
        model = tk.spectrumKNN(x_train, y_train, i)
        index.append(i)
        y_predict = model.predict(x_test)
        print(y_predict)
        sum = len(y_predict)
        correct = 0
        for j in range(len(y_predict)):
            if y_predict[j] == y_test[j]:
                correct = correct + 1
        accuracies.append(correct/sum)

        #print(index)
        print('accuracy:' + str(correct / sum))

    #print(max(accuracies))
    #for i in range(len(accuracies)):
    for i in range(len(accuracies)):
        if accuracies[i]==max(accuracies):
            print(accuracies[i])
            print('max：'+str(index[i]))
    fig, ax = plt.subplots()
    ax.set_xlabel('K value')
    ax.set_ylabel('accuracy')
    ax.plot(index,accuracies, 'b', label='KNN', linestyle='dashdot')
    ax.legend()
    plt.show()
    #scores = cross_val_score(model, x_train, y_train, cv=20, scoring='accuracy')
    #n_scores.append(scores.mean())
    #predict=model.predict([intensity[3]])
    #print(polymerName)

    #print(scores)
    # print(model.score(x_test, y_test))
    # print(model.predict(x_test))
