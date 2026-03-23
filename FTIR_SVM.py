import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from math import isnan
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import  svm
from sklearn.model_selection import train_test_split
from utils import utils
from tensorflow.keras.models import load_model
from scipy.signal import savgol_filter
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from PLS import airPLS
from FTIR_GAN2 import GAN
class TestSVM:
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
    def plot_confusion_matrix(cm, labels_name, title):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
        plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
        plt.title(title)  # 图像标题
        plt.colorbar()
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 20,
                 }
        num_local = np.array(range(len(labels_name)))
        plt.xticks(num_local, labels_name, rotation=90, size=15)  # 将标签印在x轴坐标上
        plt.yticks(num_local, labels_name, size=15)  # 将标签印在y轴坐标上
        plt.ylabel('True label', font2)
        plt.xlabel('Predicted label', font2)
        plt.show()

    def spectrumSVM(x,y,cvalue,kModel,decFunction):
        model=svm.SVC(C=cvalue, kernel=kModel, decision_function_shape=decFunction)

        model.fit(x,y)

        return model
from FTIR_deriv import cal_deriv,cal_2nd_deriv,cal_3rd_deriv
if __name__ == '__main__':

    tS=TestSVM
    #this is for the 216_2018.csv
    #polymerName, polymerID, waveLength, intensity=tk.parseData2('216_2018_1156_MOESM5_ESM.csv',3,1179)
    #this is for the D4 public_csv
    #polymerName, waveLength, intensity,polymerID,x_each,y_each= utils.parseData2('D4_4_publication5.csv', 2, 1763)
    # pList = list(set(polymerName))
    polymerName, waveLength, intensity, polymerID = utils.parseData3rd('dataset/216_2018_1156_MOESM2_ESM modified.csv')
    PN=[]
    for item in polymerName:
        if item  not in PN:
            PN.append(item)

    x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.7, random_state=0)
    accuracies=[]
    # for i in range(len(x_train)):
    #     x_train[i, :] = savgol_filter(x_train[i, :], 45, 2, mode='nearest')
    # PN = utils.getPN('D4_4_publication5.csv')
    y_add = []
    # for i in range(len(PN)):
    #     latent_dim = 100
    #     numSynth = 1
    #     generatorPS = load_model('selected'+PN[i]+" Generateor 50000")
    #     random_latent_vectors = tf.random.normal(shape=(numSynth, latent_dim))
    #     generated_specs_ps = generatorPS(random_latent_vectors)
    #     print(generated_specs_ps.shape)
    #     generated_specs_ps = generated_specs_ps.numpy()
    #     data = generated_specs_ps.reshape((numSynth, 1761))
    #     dataBaseline = []
    #
    #     for item in data:
    #         item = item - airPLS(item)
    #         dataBaseline.append(item)
    #
    #     # def normalize(arr: np.ndarray) -> np.ndarray:
    #     #     arr -= arr.min()
    #     #     arr /= arr.max()
    #     #     return arr
    #     data = normalize(dataBaseline, 'max')
    #     for j in range(numSynth):
    #         data[j, :] = savgol_filter(data[j, :], 45, 2, mode='nearest')
    #
    #     for k in range(len(data)):
    #         y_add.append(i)
    #     #print(y_add)
    #
    #     x_train = np.vstack((x_train, data))
    #     # gan = GAN(1761, x_train)
    #     # data = gan.train(epochs=200, batch_size=32, sample_interval=1000)
    #     # print(data)
    #     # indicesPS = [i for i, id in enumerate(y_train) if id == 0]
    # y_add=np.array(y_add)
    # y_train = np.concatenate((y_train, y_add), axis=0)
    # latent_dim = 100
    # numSynth = 200
    # generatorPS = load_model("Poly(ethylene)fouling Generator")
    # random_latent_vectors = tf.random.normal(shape=(numSynth, latent_dim))
    # generated_specs_ps = generatorPS(random_latent_vectors)
    # print(generated_specs_ps.shape)
    # generated_specs_ps = generated_specs_ps.numpy()
    # data = generated_specs_ps.reshape((numSynth, 1761))
    # dataBaseline = []
    # for item in data:
    #     item = item - airPLS(item)
    #     dataBaseline.append(item)
    #
    # # def normalize(arr: np.ndarray) -> np.ndarray:
    # #     arr -= arr.min()
    # #     arr /= arr.max()
    # #     return arr
    # data = normalize(dataBaseline,'max')
    # for i in range(numSynth):
    #
    #     data[i, :] = savgol_filter(data[i, :], 45, 2, mode='nearest')
    #
    # # gan = GAN(1761, x_train)
    # # data = gan.train(epochs=200, batch_size=32, sample_interval=1000)
    # # print(data)
    # # indicesPS = [i for i, id in enumerate(y_train) if id == 0]
    # y_add=[]
    # for i in range(len(data)):
    #     y_add.append(1)
    #
    # y_add=np.array(y_add)
    # latent_dim = 100
    # numSynth2 =1
    # generatorPS = load_model("selected Poly(ethylene) like Generator50000")
    # random_latent_vectors = tf.random.normal(shape=(numSynth2, latent_dim))
    # generated_specs_ps = generatorPS(random_latent_vectors)
    # print(generated_specs_ps.shape)
    # generated_specs_ps = generated_specs_ps.numpy()
    # data2 = generated_specs_ps.reshape((numSynth2, 1761))
    # dataBaseline = []
    # for item in data2:
    #     item = item - airPLS(item)
    #     dataBaseline.append(item)
    #
    # # def normalize(arr: np.ndarray) -> np.ndarray:
    # #     arr -= arr.min()
    # #     arr /= arr.max()
    # #     return arr
    # data2 = normalize(dataBaseline, 'l2')
    # for i in range(numSynth2):
    #     data2[i, :] = savgol_filter(data2[i, :], 45, 2, mode='wrap')
    # latent_dim = 100
    # numSynth3 = 1
    # generatorPS = load_model("Poly(styrene) Generator")
    # random_latent_vectors = tf.random.normal(shape=(numSynth2, latent_dim))
    # generated_specs_ps = generatorPS(random_latent_vectors)
    # print(generated_specs_ps.shape)
    # generated_specs_ps = generated_specs_ps.numpy()
    # data3 = generated_specs_ps.reshape((numSynth2, 1761))
    # dataBaseline = []
    # for item in data3:
    #     item = item - airPLS(item)
    #     dataBaseline.append(item)
    # data3 = normalize(dataBaseline, 'l2')
    # for i in range(numSynth2):
    #     data3[i, :] = savgol_filter(data3[i, :], 45, 2, mode='wrap')
    # gan = GAN(1761, x_train)
    # data = gan.train(epochs=200, batch_size=32, sample_interval=1000)
    # print(data)
    # indicesPS = [i for i, id in enumerate(y_train) if id == 0]
    # y_add = []
    # for i in range(len(data)):
    #     y_add.append(1)
    #
    # y_add = np.array(y_add)
    # y_add2 = []
    # for i in range(len(data2)):
    #     y_add2.append(2)
    #
    # y_add2 = np.array(y_add2)
    # y_add3 = []
    # for i in range(len(data3)):
    #     y_add3.append(8)
    #
    # y_add3 = np.array(y_add3)
    # # print(y_add)
    # # print(len(intensity[0]))
    # # TestSpec = x_train[indicesPS]
    #
    # # numPP, numPS = intensity.shape[0], intensity[0].shape[0]
    # # print(numPS)
    # # # for i in range(len(x_train)):
    # # #     x_train[i, :] = normalize(x_train[i, :])
    #
    # # for i in range(numSynth):
    # #
    # #     generated_specs_ps[i, :] = savgol_filter(generated_specs_ps[i, :], 45, 2, mode='nearest')
    # index=[]
    #
    # recallScore=[]
    # x_train=np.vstack((x_train, data))
    # x_train = np.vstack((x_train, data2))
    # x_train = np.vstack((x_train, data3))
    # print(x_train.shape)
    # from sklearn.preprocessing import normalize
    #
    #x_train = normalize(x_train, 'max')

    print(y_train)
    # print(y_train.shape)
    # y_train = np.concatenate((y_train, y_add2), axis=0)
    # print(y_train.shape)
    # y_train = np.concatenate((y_train, y_add3), axis=0)
    # print(y_train.shape)
    from sklearn.neighbors import KNeighborsClassifier
    KnnClf=KNeighborsClassifier(n_neighbors=2)
    #model=KnnClf.fit(x_train,y_train)
    # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(244, 258), random_state=1)
    # model.fit(x_train,y_train)
    model=tS.spectrumSVM(x_train,y_train,0.3,'linear','ovo')
    model.fit(x_train,y_train)
    y_pre=model.predict(x_test)

    utils.printScore(y_test,y_pre)
    #PN = tS.getPN('D4_4_publication5.csv')
    t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
    #SVM_report = pd.DataFrame(t)
    #SVM_report.to_csv('SVM_report5.csv')
    cm=confusion_matrix(y_test,y_pre)
    utils.plot_confusion_matrix(cm,PN,'SVM')
    # SVM_Confusion_matrix=pd.DataFrame(cm)
    # SVM_Confusion_matrix.to_csv('SVM_CM5.csv')

    # utils.plot_confusion_matrix(cm,PN,'MLP')

#    model = tS.spectrumSVM(x_train, y_train)

    y_predict = model.predict(x_test)
        #print(y_predict)
    sum = len(y_predict)
    correct = 0
    for j in range(len(y_predict)):
        if y_predict[j] == y_test[j]:
            correct = correct + 1
    accuracies.append(correct/sum)
    confusion_matrix(y_test,y_predict,labels=PN)

    #print(max(accuracies))
    #for i in range(len(accuracies)):
    # for i in range(len(accuracies)):
    #     if accuracies[i]==max(accuracies):
    #         print(accuracies[i])
    #         #print('maxValue：'+str(index[i]))
    #
    # fig, ax = plt.subplots()
    # ax.set_xlabel('K value')
    # ax.set_ylabel('accuracy')
    # ax.scatter(1,accuracies, 'b')
    # plt.show()
    # ax.legend()
    # plt.show()
    ####
    # accuracies2 = []
    # index2 = []
    # for i in range(3,40):
    #     model = tk.spectrumKNN(x_train, y_train, i)
    #     index2.append(i)
    #     y_predict2 = model.predict(x_test)
    #     #print(y_predict)
    #     sum = len(y_predict2)
    #     correct = 0
    #     for j in range(len(y_predict2)):
    #         if y_predict2[j] == y_test[j]:
    #             correct = correct + 1
    #     accuracies2.append(correct/sum)
    #
    #     #print(index)
    #     print('accuracy2:' + str(correct / sum))
    #
    # #print(max(accuracies))
    # #for i in range(len(accuracies)):
    # for i in range(len(accuracies2)):
    #     if accuracies2[i]==max(accuracies):
    #         print(accuracies2[i])
    #         print('maxValue2：'+str(index[i]))
    # #fig, ax = plt.subplots()
    # ax.set_xlabel('K value')
    # ax.set_ylabel('accuracy')
    # ax.plot(index2,accuracies2, 'r', label='KNN', linestyle='dotted')
    # #ax.legend()
    # ax.legend()
    # plt.show()
    #scores = cross_val_score(model, x_train, y_train, cv=20, scoring='accuracy')
    #n_scores.append(scores.mean())
    #predict=model.predict([intensity[3]])
    #print(polymerName)

    #print(scores)
    # print(model.score(x_test, y_test))
    # print(model.predict(x_test))
