from keras.layers import Dense, Input
from keras.models import Model
import numpy as np
import pandas  as  pd
import matplotlib.pyplot as plt
from sklearn.neural_network import  MLPClassifier
import  torch
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
MMScaler = MinMaxScaler()
from utils import utils
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from math import isnan
from sklearn.model_selection import cross_val_score
from sklearn import  svm
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, Conv1D, MaxPool1D
from keras.optimizer_v1 import SGD
from sklearn import metrics
from keras.utils import np_utils

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
    def parseData2(fileName,begin,end):

        dataset = pd.read_csv(fileName, header=None, encoding='latin-1',keep_default_na=False)
        polymerName=dataset.iloc[1:971,1]
        waveLength=dataset.iloc[0,begin:end]
        intensity=dataset.iloc[1:971,begin:end]
        polymerName= np.array(polymerName)
        waveLength=np.array(waveLength)
        #polymerNameList = {}.fromkeys(polymerName).keys()
        polymerNameList=[]
        pList= []
        for item in polymerName:
            if item not in pList:
                pList.append(item)

        polymerNameList=np.array(pList)
        polymerNameID=[]
        for i in range(len(polymerNameList)):
            polymerNameID.append(i+1)
        polymerNameData=[]
        for item in polymerName:
            for i in range(len(pList)):
                if item==pList[i]:
                    polymerNameData.append(i)


        intensity =np.array(intensity)
        for item in intensity:

            for i in range(len(item)):
                if item[i] == '':
                    item[i] = 0
        intensity = MMScaler.fit_transform(intensity)
        return polymerName,waveLength,intensity,polymerNameData
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
if __name__ == '__main__':

    tS=TestKeras
    #this is for the 216_2018.csv
    #polymerName, polymerID, waveLength, intensity=tk.parseData2('216_2018_1156_MOESM5_ESM.csv',3,1179)
    #this is for the D4 public_csv
    polymerName, waveLength, intensity,polymerID,x_each,y_each= utils.parseData2('D4_4_publication5.csv', 2, 1763)
    pList = list(set(polymerName))
    print(max(polymerID))
    x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=1)
    input_img=1761
    encoding_dim=256
    input_img = Input(shape=(1761,))
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    #x_train, x_test, y_train1, y_test1 = map(torch.tensor, (x_train, x_test, y_train, y_test))

    # encoded = Dense(1024, activation='relu')(input_img)
    # encoded = Dense(512, activation='relu')(encoded)
    # encoded = Dense(512, activation='relu')(encoded)
    # encoder_output = Dense(encoding_dim)(encoded)
    #
    # decoded = Dense(512, activation='relu')(encoder_output)
    # decoded = Dense(512, activation='relu')(decoded)
    # decoded = Dense(1024, activation='relu')(decoded)
    # decoded = Dense(1761, activation='tanh')(decoded)
    # autoencoder = Model(input=input_img, output=decoded)  # 实例化
    # encoder = Model(input=input_img, output=encoder_output)
    #
    # autoencoder.compile(
    #     optimizer='adam',
    #     loss='mse'
    # )
    # autoencoder.fit(x_train,x_train,
    #                 epochs = 20,
    #                 batch_size = 256,
    #                 shuffle = True)
    #
    # print(encoder.predict(x_test))
    # layer_model = Model(inputs=autoencoder.input, outputs=autoencoder.layers[4].output)
    PN = tS.getPN('D4_4_publication5.csv')
    nb_features=1761
    nb_class=len(PN)
    print(nb_class)
    x_train=x_train.reshape(-1,1761,1)
    x_test = x_test.reshape(-1, 1761, 1)
    print(x_train.shape)
    # X_train_r = np.zeros((len(x_train), nb_features, 3))
    # X_train_r[:, :, 0] = x_train[:, :nb_features]
    # X_train_r[:, :, 1] = x_train[:, nb_features:1024]
    # X_train_r[:, :, 2] = x_train[:, 1024:]
    # X_train_r = np.zeros((len(x_train), 1, 3))
    # X_train_r[:, :, 0] = x_train[:, :nb_features]
    # X_train_r[:, :, 1] = x_train[:, nb_features:1024]
    # X_train_r[:, :, 2] = x_train[:, 1024:]
    #feature = layer_model.predict(x_train)
    #x_test_feature= layer_model.predict(x_test)
    dense_num = 6
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_shape=(1761, 1), padding="same"))
    # 池化层1
    model.add(MaxPool1D(pool_size=3, strides=3))

    # 卷积层2 + relu
    model.add(Conv1D(64, 3, strides=1, activation='relu', padding='same'))
    # 池化层2
    model.add(MaxPool1D(pool_size=3, strides=3))

    # 神经元随机失活
    model.add(Dropout(0.25))
    # 拉成一维数据
    model.add(Flatten())
    # 全连接层1
    model.add(Dense(1024))
    # 激活层
    model.add(Activation('relu'))

    # 随机失活
    model.add(Dropout(0.4))
    # 全连接层2
    model.add(Dense(12))
    # Softmax评分
    model.add(Activation('softmax'))

    # 查看定义的模型
    model.summary()

    # 自定义优化器参数
    # rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # lr表示学习速率
    # decay是学习速率的衰减系数(每个epoch衰减一次)
    # momentum表示动量项
    # Nesterov的值是False或者True，表示使不使用Nesterov momentum
    sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)

    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 训练
    # history = model.fit(x_train, y_train, epochs=50, batch_size=16,
    #                     verbose=1, validation_data=[x_test, y_test])
    model.fit(x_train, y_train, epochs=50, batch_size=16,
                        verbose=1)


    # model.fit(x_train, y_train,
    #           epochs=nb_epoch,
    #           batch_size=16,
    #           shuffle=True)
    #model.load_weights('model/Conv1D_FTIR_model.h5')
    y_pre=model.predict(x_test)

    print(y_pre)
    # model = svm.SVC(C=1, kernel='linear', decision_function_shape='ovo')
    # model.fit(feature,y_train)
    # modelMLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 128), random_state=1)
    # modelMLP.fit(feature, y_train)
    # modelSVM = svm.SVC(C=1, kernel='linear', decision_function_shape='ovo')
    # modelKNN = KNeighborsClassifier(n_neighbors=5)
    # modelRF = RandomForestClassifier(n_jobs=-1)
    # modelRF.fit(feature, y_train)
    # modelKNN.fit(feature, y_train)
    # modelSVM.fit(feature, y_train)
    # SVMPre = modelSVM.predict(x_test_feature)
    # RFPre = modelRF.predict(x_test_feature)
    # KNNPre = modelKNN.predict(x_test_feature)
    # MLPPre = modelMLP.predict(x_test_feature)
    # print('MLP:'+str(modelMLP.score(x_test_feature,y_test)))
    # print('SVM:'+str(modelSVM.score(x_test_feature,y_test)))
    # print('KNN:' + str(modelKNN.score(x_test_feature, y_test)))
    # print('RF:' + str(modelRF.score(x_test_feature, y_test)))
    # PN = tS.getPN('D4_4_publication5.csv')
    # # t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
    # # SVM_report = pd.DataFrame(t)
    # # SVM_report.to_csv('MLP_report5.csv')
    # MLPcm = confusion_matrix(y_test, MLPPre)
    # tS.plot_confusion_matrix(MLPcm, PN, 'AEC-MLP')
    # SVMcm = confusion_matrix(y_test, SVMPre)
    # tS.plot_confusion_matrix(SVMcm, PN, 'AEC-SVM')
    # KNNcm = confusion_matrix(y_test, KNNPre)
    # tS.plot_confusion_matrix(KNNcm, PN, 'AEC-KNN')
    # RFcm = confusion_matrix(y_test, RFPre)
    # tS.plot_confusion_matrix(RFcm, PN, 'AEC-RF')
    # y_predict = model.predict(x_test_feature)
    # # print(y_predict)
    # sum = len(y_predict)
    # correct = 0
    # accuracies = []
    # for j in range(len(y_predict)):
    #     if y_predict[j] == y_test[j]:
    #         correct = correct + 1
    #     accuracies.append(correct / sum)
    #
    # # print(index)
    # print('accuracy:' + str(correct / sum))
    model_file='model/Conv1D_FTIR_model.h5'
    model.save(model_file)
    y_pre_index=[]
    for i in range(len(y_pre)):
        y_pre_index.append(np.argmax(y_pre[i]))
    y_pre_index2=np.argmax(y_pre,axis=1)
    print(y_pre_index2)
    utils.printScore(y_test,y_pre_index)
    cm = confusion_matrix(y_test, y_pre_index)

    utils.plot_confusion_matrix(cm, PN, 'MLP')

    print(metrics.r2_score(y_test,y_pre_index))