from keras.layers import Input,add
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Model
import numpy as np
import pandas  as  pd
import matplotlib.pyplot as plt

from sklearn.neural_network import  MLPClassifier
import  torch
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K
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
    def plot_confusion_matrix(cm, labels_name, title):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
        plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
        plt.title(title)  # 图像标题
        plt.colorbar()
        num_local = np.array(range(len(labels_name)))
        plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
        plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def parseData2(fileName, begin, end):
        dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False)
        polymerName = dataset.iloc[1:971, 1]
        waveLength = dataset.iloc[0, begin:end]
        intensity = dataset.iloc[1:971, begin:end]
        polymerName = np.array(polymerName)
        waveLength = np.array(waveLength)
        polymerNameID = []
        PN = []
        for item in polymerName:
            if item not in PN:
                PN.append(item)
        PN = np.array(PN)
        for i in range(len(PN)):
            polymerNameID.append(i + 1)
        polymerNameData = []
        for item in polymerName:
            for i in range(len(PN)):
                if item == PN[i]:
                    polymerNameData.append(i + 1)

        intensity = np.array(intensity)
        intensity = MMScaler.fit_transform(intensity)
        return polymerName, waveLength, intensity, polymerNameData
    def spectrumSVM(x,y,cvalue,kModel,decFunction):
        model=svm.SVC(C=cvalue, kernel=kModel, decision_function_shape=decFunction)

        model.fit(x,y)

        return model

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
class LSTMAugmentation(object):
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
        for m in range(len(self.x_train)):

            for n in range(len(self.x_train)):
                data.append([self.x_train[m],self.x_train[n]])
                data3.append(self.x_train[n])
                data2.append(self.x_train[m])

                rm=randomText(len(self.x_train), m, n)

                print(len(self.x_train),m,n,rm)
                data4.append(self.x_train[rm])

                ylabel.append(numPoly)

            #data=np.array(data,dtype=np.float32)
            #print(data.shape)

        ylabel=np.array(ylabel)

        data=np.array(data,dtype=float)
        data2=np.array(data2,dtype=float)
        data3=np.array(data3,dtype=float)
        data4 = np.array(data4, dtype=float)
        inputs = Input(shape=(1, 1761))
        encoded = LSTM(1024)(inputs)
        decoded = RepeatVector(1)(encoded)
        decoded = LSTM(1761, return_sequences=True)(decoded)

        autoencoder = Model(inputs=inputs, outputs=decoded)  # 实例化
        autoencoder.compile(
            optimizer='adam',
            loss='mse'
        )
        data2 = data2.reshape((data2.shape[0], 1, data2.shape[1]))
        autoencoder.fit(data2,data2,
                        epochs = 30,
                        batch_size = 32,
                        shuffle = True)

        intensity0=autoencoder.predict(data2)
        intensity0 = intensity0.reshape((intensity0.shape[0], intensity0.shape[2]))
        return intensity0,ylabel
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
if __name__ == '__main__':

    tS=TestKeras
    #this is for the 216_2018.csv
    #polymerName, polymerID, waveLength, intensity=tk.parseData2('216_2018_1156_MOESM5_ESM.csv',3,1179)
    #this is for the D4 public_csv
    polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData2('D4_4_publication5.csv', 2, 1763)
    pList = utils.getPN('D4_4_publication5.csv')
    statisticForMetrics = []
    model = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
    models = [model]
    cmTotal = np.zeros((12, 12))

    v = 0
    t_report = []
    scoreTotal = np.zeros(5)
    for itr in range(1,2):
        x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.6, random_state=itr)
        input_img=1761
        encoding_dim=512
        input_img = Input(shape=(1761,))
        intensity=np.array(intensity,dtype=np.float32)
        x_train = np.array(x_train, dtype=np.float32)
        print(x_train.shape)
        PN = utils.getPN('D4_4_publication5.csv')
        y_e = []
        for it in y_train:
            if it not in y_e:
                y_e.append(it)
        if len(y_e) < 12:
            continue
        y_e2 = []
        for itt in y_test:
            if itt not in y_e2:
                y_e.append(itt)
        if len(y_e) < 12:
            continue
        datas=[]
        for i in range(len(PN)):

            indicesPS = [j for j, id in enumerate(y_train) if id == i]
            data=x_train[indicesPS]
            datas.append(data)
        #datas=np.array(datas)
        # for item in datas:
        #     print(item.shape)
        #print(datas.shape)
        dataSquare=[]
        dataSquare2 = []
        dataSquare3=[]
        dataSquare4 = []
        numPoly=0
        ylabel = []
        for item in datas:


            data = []
            data2=[]
            data3=[]
            data4=[]
            for m in range(len(item)):

                for n in range(len(item)):
                    data.append([item[m],item[n]])
                    data3.append(item[n])
                    data2.append(item[m])

                    rm=randomText(len(item), m, n)

                    print(len(item),m,n,rm)
                    data4.append(item[rm])

                    ylabel.append(numPoly)
            numPoly += 1
            #data=np.array(data,dtype=np.float32)
            #print(data.shape)
            dataSquare.append(data)
            dataSquare2.append(data2)
            dataSquare3.append(data3)
            dataSquare4.append(data4)
        ylabel=np.array(ylabel)


        #dataSquare=np.array(dataSquare)
        #print('1111111111111111',dataSquare.shape)

        x_test = np.array(x_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)
        # # x_train, x_test, y_train1, y_test1 = map(torch.tensor, (x_train, x_test, y_train, y_test))
        # print(x_train,x_test)
        for i in range(len(datas)):
            datas[i] = datas[i].astype('float64')
        dataSquare=np.array(dataSquare)
        dataSquare2 = np.array(dataSquare2)
        dataSquare3 = np.array(dataSquare3)
        print('dataSquare2 shape',dataSquare2.shape)
        print('dataSquare3 shape', dataSquare3.shape)
        for i in range(len(dataSquare)):
            dataSquare[i] = np.array(dataSquare[i],dtype=float)
        for i in range(len(dataSquare2)):
            dataSquare2[i] = np.array(dataSquare2[i],dtype=float)
        for i in range(len(dataSquare3)):
            dataSquare3[i] = np.array(dataSquare3[i],dtype=float)
        for i in range(len(dataSquare4)):
            dataSquare4[i] = np.array(dataSquare4[i],dtype=float)
        # for i in range(len(datas)):
        #     datas[i] = datas[i].astype('float64')
        #y_train = np.concatenate((y_train, y_add), axis=0)
        #x_train, x_test, y_train1, y_test1 = map(torch.from_numpy, (x_train, x_test, y_train, y_test))
        numForSquare=0
        for i in range(7,8):
            print('iiiiiiiiiiiiii',len(dataSquare2[i]))

            if len(datas[i])>=100:
                continue
            numForSquare +=1

            # input1 = Input(shape=(1761,))
            # input2 = Input(shape=(1761,))
            # encoded1 = Dense(1761, activation='relu')(input1)
            # encoded2= Dense(1761, activation='relu')(input2)
            # #d0 = tf.keras.layers.concatenate([encoded1, encoded2], axis=1)
            timesteps=1761
            # d0 = add([encoded1, encoded2])
            # inputs = Input(shape=(1761, 1))
            # encoded = Dense(1024, activation='relu')(d0)
            # input2 = Lambda(lambda input2: K.expand_dims(input2, axis=-1))(input2)
            latent_dim=512

            # model = Sequential()
            # model.add(LSTM(1024, activation='relu', input_shape=(1761, 1)))
            # model.add(RepeatVector(timesteps))
            # model.add(TimeDistributed(Dense(1)))
            # model.compile(optimizer='adam', loss='mse')
            # inputs = Input(shape=(timesteps, 1))
            # encoded = LSTM(1024, activation='relu')(inputs)
            #
            # decoded = RepeatVector(timesteps)(encoded)
            # decoded = LSTM(1024, activation='relu')(decoded)
            #
            # sequence_autoencoder = Model(inputs, decoded)
            # model = Model(inputs, encoded)
            inputs = Input(shape=(1, 1761))
            encoded = LSTM(1024)(inputs)
            decoded = RepeatVector(1)(encoded)
            decoded = LSTM(1761, return_sequences=True)(decoded)

            model = Model(inputs, decoded)
            encoder = Model(inputs, encoded)
            # encoder = Model(inputs, encoded)
            # encoded = Dense(1024, activation='relu')(d0)
            # # encoded = Dense(512, activation='relu')(encoded)
            # # encoder_output = Dense(encoding_dim)(encoded)
            #
            # decoded = Dense(512, activation='relu')(encoded)
            # # decoded = Dense(512, activation='relu')(decoded)
            # decoded = Dense(1024, activation='relu')(decoded)
            # decoded = Dense(1761, activation='sigmoid')(decoded)
            # autoencoder = Model(inputs=[input1,input2], outputs=decoded)  # 实例化
            # # encoder = Model(inputs=[input1,input2], outputs=encoder_output)
            # def mycrossentropy(y_true, y_pred, e=0.1):
            #     loss1 = y_pred
            #     loss2 = K.categorical_crossentropy(K.ones_like(y_pred) , y_pred)
            #     return (1 - e) * loss1 + e * loss2
            model.compile(
                optimizer='adam',
                loss='mse'
            )
            print(dataSquare2[i].shape[0],dataSquare2[i].shape[1])
            dataSquare2[i]=dataSquare2[i].reshape((dataSquare2[i].shape[0],1,dataSquare2[i].shape[1]))
            dataSquare4[i] = dataSquare4[i].reshape((dataSquare4[i].shape[0], 1, dataSquare4[i].shape[1]))
            datas[i]=datas[i].reshape((datas[i].shape[0],1,datas[i].shape[1]))
            # dataSquare4[i] = dataSquare4[i].reshape((dataSquare4[i].shape[0], dataSquare4[i].shape[1], 1))
            model.fit( dataSquare2[i],dataSquare2[i],
                            epochs = 50,
                            batch_size = 32,
                            shuffle = True)
            # encoder.predict([dataSquare2[i],dataSquare3[i]])
            # print(encoder.predict([dataSquare2[i],dataSquare3[i]]))
            indicesPS = [j for j, id in enumerate(ylabel) if id == i]
            ylabel0 = ylabel[indicesPS]
            count=200-len(dataSquare2[i])
            if count>=len(dataSquare2[i]):
                count=len(dataSquare2[i])
            intensity0=model.predict(dataSquare2[i])
            np.random.shuffle(intensity0)
            # intensityShuffle=np.array(intensityShuffle)
            # print(intensityShuffle)
            intensityPlus=intensity0[0:count]
            ylabel0=ylabel0[0:count]
            print('augmentation intensity shape',intensity0.shape)
            print('augmentation intensityPLus shape', intensityPlus.shape)
            intensityPlus=intensityPlus.reshape((intensityPlus.shape[0],intensityPlus.shape[2]))
            x_train = np.concatenate((x_train, intensityPlus), axis=0)
            y_train = np.concatenate((y_train, ylabel0), axis=0)
            print('x_train',x_train.shape)
        PN = utils.getPN('D4_4_publication5.csv')
        model = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
        #model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 128), random_state=1)
        model.fit(x_train, y_train)



        y_pre = model.predict(x_test)
        y_p = []
        for y_p_e in y_pre:
            if y_p_e not in y_p:
                y_p.append(y_p_e)
        if len(y_p) < 12:
            continue
        utils.printScore(y_test, y_pre)
        #

        t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
        cm = confusion_matrix(y_test, y_pre)
        # model = tS.spectrumSVM(x_train, y_train, 0.3, 'linear', 'ovo')
        y_pre = model.predict(x_test)

        cm = confusion_matrix(y_test, y_pre)
        cmtemp = np.zeros((12, 12))
        if len(cm) == 11:
            print('cm equals 11', len(cm))
            continue
            # for i in range(len(cm)):
            #     cmtemp[i]=np.append(cm[i],[0],axis=0)
            #     # cmtemp[i]+=cm[i]
            # cmtemp[11,11]=2
            # print('tempCM；',cmtemp)
            # cmTotal+=cmtemp
            # continue
        cmTotal = cmTotal + cm
        scores = utils.printScore(y_test, y_pre)
        v += 1
        scoreTotal += scores
        print('v:',v)
        print('numForSquare',numForSquare)
        t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
        utils.mkdir(model.__class__.__name__ + 'LSTM_square_report')
        SVM_report = pd.DataFrame(t)
        print(SVM_report)
        SVM_report.to_csv(
            model.__class__.__name__ + 'LSTM_square_report/' + model.__class__.__name__ + 'LSTM_square_report' + str(i) + '.csv')
        t_report += t
        print(t.__class__)
    print(scoreTotal / v)
    statisticForMetrics.append(scoreTotal / v)

    cmTotal = cmTotal / v
    print(cmTotal / v)

    # utils.plot_confusion_matrix(cmTotal, PN, item.__class__.__name__+'_PCA')
    utils.plot_confusion_matrix(cmTotal, PN, 'LSTM argumentation 0.6')
    statisticForMetrics = pd.DataFrame(statisticForMetrics)
    statisticForMetrics.to_csv('squareargumentation.csv')