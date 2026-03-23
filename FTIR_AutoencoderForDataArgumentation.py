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
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from math import isnan
from sklearn.model_selection import cross_val_score
from sklearn import  svm
from sklearn.model_selection import train_test_split
from utils import utils
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
if __name__ == '__main__':

    tS=TestKeras
    #this is for the 216_2018.csv
    #polymerName, polymerID, waveLength, intensity=tk.parseData2('216_2018_1156_MOESM5_ESM.csv',3,1179)
    #this is for the D4 public_csv
    polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData2('D4_4_publication5.csv', 2, 1763)
    pList = utils.getPN('D4_4_publication5.csv')

    x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=1)
    input_img=1761
    encoding_dim=512
    input_img = Input(shape=(1761,))
    intensity=np.array(intensity,dtype=np.float32)
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    # x_train, x_test, y_train1, y_test1 = map(torch.tensor, (x_train, x_test, y_train, y_test))
    print(x_train,x_test)
    #x_train, x_test, y_train1, y_test1 = map(torch.from_numpy, (x_train, x_test, y_train, y_test))
    encoded = Dense(1024, activation='relu')(input_img)
    encoded = Dense(512, activation='relu')(encoded)
    encoded = Dense(512, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    decoded = Dense(512, activation='relu')(encoder_output)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(1024, activation='relu')(decoded)
    decoded = Dense(1761, activation='tanh')(decoded)
    autoencoder = Model(inputs=input_img, outputs=decoded)  # 实例化
    encoder = Model(inputs=input_img, outputs=encoder_output)

    autoencoder.compile(
        optimizer='adam',
        loss='mse'
    )
    autoencoder.fit(x_train,x_train,
                    epochs = 20,
                    batch_size = 256,
                    shuffle = True)




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
    PN = tS.getPN('D4_4_publication5.csv')
    import random

    statisticForMetrics = []
    model = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
    models = [model]
    cmTotal = np.zeros((12, 12))

    m = 0
    t_report = []
    scoreTotal = np.zeros(5)
    for i in range(30):
        model = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
        x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=i)
        waveLength = np.array(waveLength, dtype=np.float)
        PN = utils.getPN('D4_4_publication5.csv')
        y_add = []
        data_plot = []
        y_e=[]
        for it in y_train:
            if it not in y_e:
                y_e.append(it)
        print(y_e)
        if len(y_e)<12 :
            continue
        encoded = Dense(1024, activation='relu')(input_img)
        encoder_output = Dense(encoding_dim)(encoded)

        decoded = Dense(1024, activation='relu')(encoder_output)

        decoded = Dense(1761, activation='tanh')(decoded)
        autoencoder = Model(inputs=input_img, outputs=decoded)  # 实例化
        encoder = Model(inputs=input_img, outputs=encoder_output)

        autoencoder.compile(
            optimizer='adam',
            loss='mse'
        )
        autoencoder.fit(x_train, x_train,
                        epochs=30,
                        batch_size=32,
                        shuffle=True)

        print(encoder.predict(x_test))
        layer_model = Model(inputs=autoencoder.input, outputs=autoencoder.outputs)
        for n in range(3):
            data = layer_model.predict(x_train)

                # data_plot=np.vstack((data_plot,data))
            x_train = np.concatenate((x_train, data), axis=0)

            data_plot = np.array(data_plot)
            from sklearn.preprocessing import normalize
            y_add=y_train
            # x_train=normalize(x_train,'max')
            y_train = np.concatenate((y_train, y_add), axis=0)
        # from sklearn.neighbors import KNeighborsClassifier
        # from sklearn.metrics import classification_report
        # from sklearn.metrics import confusion_matrix
        from sklearn.neural_network import MLPClassifier
        # from sklearn import svm
        #
        # KnnClf = KNeighborsClassifier(n_neighbors=2)
        # model=KnnClf.fit(x_train,y_train)
        # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(244, 258), random_state=1)
        # model.fit(x_train, y_train)

        model.fit(x_train, y_train)
        y_pre = model.predict(x_test)
        y_p=[]
        for y_p_e in y_pre:
            if y_p_e not in y_p:
                y_p.append(y_p_e)
        if len(y_p)<12:
            continue
        utils.printScore(y_test, y_pre)
        PN = utils.getPN('D4_4_publication5.csv')
        t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
        SVM_report = pd.DataFrame(t)
        SVM_report.to_csv('SVM_report5.csv')
        cm = confusion_matrix(y_test, y_pre)
        #model = tS.spectrumSVM(x_train, y_train, 0.3, 'linear', 'ovo')
        y_pre = model.predict(x_test)

        cm = confusion_matrix(y_test, y_pre)
        cmtemp=np.zeros((12,12))
        if len(cm)==11:
            continue
            # for i in range(len(cm)):
            #     cmtemp[i]=np.append(cm[i],[0],axis=0)
            #     # cmtemp[i]+=cm[i]
            # cmtemp[11,11]=2
            # print('tempCM；',cmtemp)
            # cmTotal+=cmtemp
            # continue
        cmTotal=cmTotal+cm
        scores=utils.printScore(y_test, y_pre)
        m+=1
        scoreTotal+=scores
        print(m)

        t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
        utils.mkdir(model.__class__.__name__+'AE_plus_report')
        SVM_report = pd.DataFrame(t)
        print(SVM_report)
        SVM_report.to_csv(model.__class__.__name__+'AE_plus_report/'+model.__class__.__name__ + 'AE_plus_report'+str(i) + '.csv')
        print(t.__class__)
        #t=np.array(t)
        print('Report: -----',t)
        t_report+=t
        print(t.__class__)
    print(scoreTotal/m)
    statisticForMetrics.append(scoreTotal/m)

    cmTotal=cmTotal/m
    print(cmTotal/m)

    # utils.plot_confusion_matrix(cmTotal, PN, item.__class__.__name__+'_PCA')
    utils.plot_confusion_matrix(cmTotal, PN, 'Autoencoder argumentation')
    statisticForMetrics=pd.DataFrame(statisticForMetrics)
    statisticForMetrics.to_csv('statisticForMetricsPCA.csv')