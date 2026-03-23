import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams
import numpy as np
from utils import utils
from sklearn.model_selection import train_test_split
import random
from sklearn import svm
from sklearn.metrics import confusion_matrix
polymerName, waveLength, intensity, polymerID=utils.parseDataForSecondDataset('new_SecondDataset2.csv')

waveLength = np.array(waveLength)
if waveLength[0] > waveLength[-1]:
    rng = waveLength[0] - waveLength[-1]
else:
    rng = waveLength[-1] - waveLength[0]
half_rng = rng / 2
normalized_wns = (waveLength - np.mean(waveLength)) / half_rng
for i in range(len(intensity)):
    for m in range(len(intensity[i])):
        if  intensity[i][m]<=0:
            intensity[i][m]=0.01
# result = seasonal_decompose(intensity, model='multiplicative', period=4)

# rcParams['figure.figsize'] = 10, 5
# result.plot()
# plt.plot(normalized_wns,result.resid[0])
# plt.figure(figsize=(40, 10))
# plt.show()
# plt.clf()
# import seaborn as sns
# import matplotlib.pyplot as plt
# from statsmodels.tsa.seasonal import STL
#
#
# stl = STL(intensity[0], period=12, robust=True)
# res_robust = stl.fit()
# print('resid',res_robust.resid)
# print(res_robust)
# plt.plot(normalized_wns,res_robust.resid)

cmTotal = np.zeros((4, 4))
m = 0
t_report = []
scoreTotal = np.zeros(5)
for seedNum in range(20):
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
    if len(pID) <= 3:
        continue
    # for n in range(len(PN)):
    #     numSynth = 2
    #     indicesPS = [l for l, id in enumerate(y_train) if id == n]
    #     intensityForLoop = x_train[indicesPS]
    #     datas.append(intensityForLoop)
    #     datas2.append(intensityForLoop)
    # for itr in range(len(PN)):
    #     _, coefs_ = emsc(
    #         datas[itr], waveLength,reference=None,
    #         order=2,
    #         return_coefs=True)
    #
    #     coefs_std = coefs_.std(axis=0)
    #     indicesPS = [l for l, id in enumerate(y_train) if id == itr]
    #     label=y_train[indicesPS]
    #
    #
    #     reference=datas[itr].mean(axis=0)
    #     emsa = EMSA(coefs_std, waveLength, reference, order=2)
    #
    #     generator = emsa.generator(datas[itr], label,
    #         equalize_subsampling=False, shuffle=False,
    #         batch_size=200)
    #     augmentedSpectrum = []
    #     for i, batch in enumerate(generator):
    #         if i >2:
    #             break
    #         augmented = []
    #         for augmented_spectrum, label in zip(*batch):
    #
    #             plt.plot(waveLength, augmented_spectrum, label=label)
    #             augmented.append (augmented_spectrum)
    #         augmentedSpectrum.append(augmented)
    #         # plt.gca().invert_xaxis()
    #         # plt.legend()
    #         # plt.show()
    #     augmentedSpectrum=np.array(augmentedSpectrum)
    #     y_add=[]
    #     for item in augmentedSpectrum[0]:
    #         y_add.append(itr)
    #     from sklearn.preprocessing import normalize
    #     augmentedSpectrum[0] = normalize(augmentedSpectrum[0], 'max')
    #     x_train=np.concatenate((x_train,augmentedSpectrum[0]),axis=0)
    #     y_train=np.concatenate((y_train,y_add),axis=0)
    for n in range(1,3):

        numSynth = 1
        indicesPS = [i for i, id in enumerate(y_train) if id == n]
        y_trainindex = y_train[indicesPS]
        intensityForLoop = x_train[indicesPS]
        referenceMean = np.mean(intensityForLoop, axis=0)

        intensityplusnoise = []
        intensityforRandomLoop = []
        idexes = np.arange(len(intensityForLoop))
        print(idexes)
        indexForIntensity = idexes.take(range(0, 20), mode='wrap')
        intensityforRandomLoop = intensityForLoop[indexForIntensity]
        peaknosieforloop = []
        result = seasonal_decompose(intensityforRandomLoop, model='multiplicative', period=4)
        resids=result.resid[2:-2]
        trends=result.trend[2:-2]
        resids=np.array(resids)
        intensityLoop=intensityforRandomLoop[2:-2]
        resids=intensityLoop-trends
        # residsnew=[]
        # for xm in range(len(intensityforRandomLoop)):
        #     for tm in range(len(trends)):
        #         residsnew.append( intensityforRandomLoop[xm]-trends[tm])
        # datanew=[]
        # for tn in range(len(trends)):
        #     for rn in range(len(residsnew)):
        #         rr = np.random.uniform(0, 50) * 0.1
        #         datanew.append(trends[tn]+residsnew[rn]*rr)
        trends=np.array(trends)
        print(resids.shape)
        print(trends.shape)
        dataforaugmentation=[]
        for q in range(len(trends)):
            for p in range(len(resids)):
                rr=np.random.uniform(0,50)*0.1
                # print('uniform',rr)
                # rr=random.randint(-50,50)*0.1
                dataforaugmentation.append(trends[q]+resids[p]*rr)

        from sklearn.preprocessing import normalize

        y_addEmsc =[]

        db = "db3"

        #print(data2D.shape)
        # dataforaugmentation=datanew
        dataforaugmentation=np.array(dataforaugmentation,dtype=np.float)

        data2D = normalize(dataforaugmentation, 'max')
        #data2D = random.sample(list(data2D), 300)
        data2D = np.array(data2D)
        print(data2D.shape)
        y_add = []
        for item in data2D:
            y_add.append(n)
        print('data2dShape', data2D.shape)
        x_train = np.concatenate((x_train, data2D), axis=0)

        y_train = np.concatenate((y_train, y_add), axis=0)
    # for n in range(len(PN)):
    #     numSynth = 2
    #     indicesPS = [l for l, id in enumerate(y_train) if id == n]
    #     intensityForLoop = x_train[indicesPS]
    #     datas.append(intensityForLoop)
    #     datas2.append(intensityForLoop)
    # for itr in range(len(PN)):
    #     _, coefs_ = emsc(
    #         datas[itr], waveLength,reference=None,
    #         order=2,
    #         return_coefs=True)
    #
    #     coefs_std = coefs_.std(axis=0)
    #     indicesPS = [l for l, id in enumerate(y_train) if id == itr]
    #     label=y_train[indicesPS]
    #
    #
    #     reference=datas[itr].mean(axis=0)
    #     emsa = EMSA(coefs_std, waveLength, reference, order=2)
    #
    #     generator = emsa.generator(datas[itr], label,
    #         equalize_subsampling=False, shuffle=False,
    #         batch_size=200)
    #     augmentedSpectrum = []
    #     for i, batch in enumerate(generator):
    #         if i >2:
    #             break
    #         augmented = []
    #         for augmented_spectrum, label in zip(*batch):
    #
    #             plt.plot(waveLength, augmented_spectrum, label=label)
    #             augmented.append (augmented_spectrum)
    #         augmentedSpectrum.append(augmented)
    #         # plt.gca().invert_xaxis()
    #         # plt.legend()
    #         # plt.show()
    #     augmentedSpectrum=np.array(augmentedSpectrum)
    #     y_add=[]
    #     for item in augmentedSpectrum[0]:
    #         y_add.append(itr)
    #     from sklearn.preprocessing import normalize
    #     augmentedSpectrum[0] = normalize(augmentedSpectrum[0], 'max')
    #     x_train=np.concatenate((x_train,augmentedSpectrum[0]),axis=0)
    #     y_train=np.concatenate((y_train,y_add),axis=0)
    # # randnum = random.randint(0, 100)
    # # random.seed(randnum)
    # random.shuffle(x_train)
    # random.seed(randnum)
    # random.shuffle(y_train)
    # x_train=xtraindata
    # y_train=ytraindata
    print(x_train.shape)
    print(y_train.shape)
    # x_train, x_test0, y_train, y_test0 = train_test_split(x_train, y_train, test_size=0.1, random_state=1)

    model = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
    model = model.fit(x_train, y_train)
    y_pre = model.predict(x_test)

    utils.printScore(y_test, y_pre)
    # for item in polymerName:
    #     if item not in PN:
    #         PN.append(item)

    # t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
    # SVM_report = pd.DataFrame(t)
    # SVM_report.to_csv('SVM_report5.csv')
    cm = confusion_matrix(y_test, y_pre)
    # utils.plot_confusion_matrix(cm,PN,'EMSA_SVM')
    scores = utils.printScore(y_test, y_pre)

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
plt.clf()
# utils.plot_confusion_matrix(cmTotal, PN, 'Split partial data agumentation_SVM')
utils.plot_confusion_matrix(cmTotal, PN, 'seasonal_decompose_SVM')
# utils.plot_confusion_matrix(cmTotal, PN, 'ESMA agumentation_SVM')
# utils.plot_confusion_matrix(cmTotal, PN, 'ESMA partial data agumentation_SVM')
fig, ax = plt.subplots(nrows=3, ncols=1)
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }
# for item in _:
#     ax[0].plot(waveLength, item, '-r')
for item in data2D:
    ax[1].plot(waveLength, item, '-r')
# for item in augmentedSpectrum[0]:
#     ax[2].plot(waveLength, item, '-r')
labels0 = ax[0].get_xticklabels() + ax[0].get_yticklabels()
labels1 = ax[1].get_xticklabels() + ax[1].get_yticklabels()
labels2 = ax[2].get_xticklabels() + ax[2].get_yticklabels()
ax[0].tick_params(labelsize=15)
ax[1].tick_params(labelsize=15)
ax[2].tick_params(labelsize=15)
# [label0.set_fontname('normal') for label0 in labels0]
# [label0.set_fontstyle('normal') for
# label0 in labels0]
ax[0].set_title('EMSC', font2)
ax[1].set_title('Original', font2)
ax[2].set_title('EMSA', font2)
plt.show()
    # plt.clf()
