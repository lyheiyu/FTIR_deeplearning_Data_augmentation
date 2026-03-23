import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import matplotlib as mpl
from Reconstruction import prepareSpecSet
from scipy.signal import savgol_filter
import os
from sklearn.decomposition import PCA
from readData import readFile
from utils import utils
latent_dim=100
numSynth = 300
import wandb
# wandb.init(name='GAN-4thdataset', project="Augmentation")

# generatorPS = load_model("Poly(ethylene) like Generator50000")
# random_latent_vectors = tf.random.normal(shape=(numSynth, latent_dim))
# generated_specs_ps = generatorPS(random_latent_vectors)
# print(generated_specs_ps.shape)
# generated_specs_ps = generated_specs_ps.numpy()
# generated_specs_ps = generated_specs_ps.reshape((300, 3551))
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
# from sklearn.preprocessing import normalize
def normalize(arr: np.ndarray) -> np.ndarray:
    arr -= arr.min()
    arr /= arr.max()
    return arr
from sklearn.model_selection import train_test_split

#polymerName, waveLength, intensity, polymerID,x_each,y_each =utils.parseData11('D4_4_publication11.csv', 2, 1763)
polymerName, waveLength, intensity, polymerID = utils.parseData4th('dataset/FourthdatasetFollp-r.csv')
#polymerName, waveLength, intensity, polymerID = utils.parseDataForSecondDataset('new_SecondDataset2.csv')
# x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.7, random_state=0)
# PN=utils.getPN('D4_4_publication11.csv')
# cmTotal = np.zeros((11, 11))
PN = []
for item in polymerName:
    if item not in PN:
        PN.append(item)
cmTotal = np.zeros((len(PN), len(PN)))
m = 0
t_report = []
scoreTotal = np.zeros(5)
waveShape=len(intensity[0])
nums=[1,2]
# font2 = {'family': 'Times New Roman',
#              'weight': 'normal',
#              'size': 30,
#              }
for nt in range(20):
    x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=nt)
    pID=[]
    for item in y_train:
        if item not in pID:
            pID.append(item)
    if len(pID) < len(PN):
        continue
    print(len(pID))
    Pidtest = []
    for item in y_test:
        if item not in Pidtest:
            Pidtest.append(item)
    print(len(Pidtest))
    if len(Pidtest) < len(PN):
        continue
    for j in range(len(PN)):
    #for j in nums:
        y_add=[]
        GeneratedModel=load_model('4th dataset GAN model/selected'+PN[j]+" Generateor 50000")
        #GeneratedModel = load_model('Second_GAN model/selected' + PN[j] + " Generateor 50000")
        #GeneratedModel = load_model('selected' + PN[j] + " Generateor 50000")
        random_latent_vectors = tf.random.normal(shape=(numSynth, latent_dim))
        generated_specs_ps = GeneratedModel(random_latent_vectors)
        print(generated_specs_ps.shape)
        generated_specs_ps = generated_specs_ps.numpy()
        generated_specs_ps = generated_specs_ps.reshape((numSynth, waveShape))
        print(generated_specs_ps.shape)
        #generated_specs_ps=normalize(generated_specs_ps, 'max')
        for i in range(len(generated_specs_ps)):
                generated_specs_ps[i, :] = normalize(generated_specs_ps[i, :])
                #generated_specs_ps[i, :] = savgol_filter(generated_specs_ps[i, :], 45, 2, mode='nearest')
        for itr in range(len(generated_specs_ps)):
            y_add.append(j)
        x_train = np.concatenate((x_train, generated_specs_ps), axis=0)

        y_train = np.concatenate((y_train, y_add), axis=0)
    #model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 64, 64), random_state=1)
    model = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
   # model=KNeighborsClassifier(n_neighbors=2)
    print('x_train',x_train.shape)
    model = model.fit(x_train, y_train)
    y_pre = model.predict(x_test)

    scores=utils.printScore(y_test, y_pre)
    # wandb.log({'scores': scores[2]})
    # PN = utils.getPN('D4_4_publication11.csv')
    t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
    SVM_report = pd.DataFrame(t)
    # SVM_report.to_csv('SVM_report5.csv')
    cm = confusion_matrix(y_test, y_pre)
    cmTotal+=cm
    scoreTotal+=scores
    m+=1
    print(m)

utils.plot_confusion_matrix(cmTotal/m, PN, "GAN SVM")
#utils.plot_confusion_matrix(cmTotal/m, PN, "GAN MLP")
print(scoreTotal/m)
indicesPS = [i for i, id in enumerate(polymerID) if id == 8]
dataset = intensity
print(len(intensity[0]))
TestSpec = intensity[indicesPS]
numPP, numPS = intensity.shape[0], intensity[0].shape[0]
print(numPS)


for i in range(len(indicesPS)):
    TestSpec[i, :] = normalize(TestSpec[i, :])
for i in range(numSynth):
    generated_specs_ps[i, :] = normalize(generated_specs_ps[i, :])
    generated_specs_ps[i, :]  = savgol_filter(generated_specs_ps[i, :] , 45, 2, mode='nearest')

for i in range(len(generated_specs_ps)):
    plt.plot(waveLength[::-1],generated_specs_ps[i] , color='blue')
font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
plt.tick_params(labelsize=30)
plt.xlim(4000, 400)
# plt.xlabel('Wavelength',font2)
# plt.ylabel('Intensity',font2)
plt.title('GAN spectral data',font2)
plt.show()


fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(131)
offset = 0
print(waveLength[::-1])
plt.tick_params(labelsize=10)
plt.xlim(4500, 400)
for i in range(5):
    offset -= 0.5
    if i == 0:
        ax1.plot(waveLength[::-1] , TestSpec[i, :] + offset, color='blue', label='Real EPR')
    else:
        ax1.plot(waveLength[::-1] , TestSpec[i , :]  + offset, color='blue')



ax1.set_title("Real Spectra")
ax1.legend()

ax2 = fig.add_subplot(132)
offset = 0
plt.tick_params(labelsize=10)
plt.xlim(4500, 400)
for i in range(5):
    offset -= 0.5
    if i == 0:
        ax2.plot(waveLength[::-1] , generated_specs_ps[i + 10, :]  + offset, color='red', label='Generated EPR')
    else:
        ax2.plot(waveLength[::-1] , generated_specs_ps[i + 10, :]  + offset, color='red')

ax2.set_title("Generated Spectra")
ax2.legend()

for ax in [ax1, ax2]:
    ax.set_xlabel("Wavenumbers (cm-1)", fontsize=12)
    ax.set_yticks([])
plt.show()

# trainPlusGenerated = np.vstack((specsPP, specsPS, generated_specs_pp, generated_specs_ps))
# pca: PCA = PCA(n_components=2, random_state=42)
# princComps: np.ndarray = pca.fit_transform(trainPlusGenerated)
#
# pcaAx: plt.Axes = fig.add_subplot(133)
# pcaAx.scatter(princComps[:numPP, 0], princComps[:numPP, 1], color='blue', s=4, label='PP Real')
# start, stop = numPP, numPP+numPS
# pcaAx.scatter(princComps[start:stop, 0], princComps[start:stop, 1], color='red', s=4, label='PS Real')
# start, stop = numPP+numPS, numPP+numPS+numSynth
# pcaAx.scatter(princComps[start:stop, 0], princComps[start:stop, 1], color='blue', alpha=0.1, label='PP generated')
# start, stop = numPP+numPS+numSynth, numPP+numPS+numSynth+numSynth
# pcaAx.scatter(princComps[start:stop, 0], princComps[start:stop, 1], color='red', alpha=0.1, label='PS generated')
# pcaAx.set_xlabel("PC 1", fontsize=12)
# pcaAx.set_ylabel("PC 2", fontsize=12)
# pcaAx.set_title("PCA Plot")
# pcaAx.legend()
#
# fig.tight_layout()