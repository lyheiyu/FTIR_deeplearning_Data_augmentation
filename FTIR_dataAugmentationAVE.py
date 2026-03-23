from keras.layers import Dense, Input,add
from keras.models import Model
import numpy as np
import keras
import pandas  as  pd
import matplotlib.pyplot as plt
from sklearn.neural_network import  MLPClassifier
import  torch
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import keras as K
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from math import isnan
from sklearn.model_selection import cross_val_score
from sklearn import  svm
from sklearn.model_selection import train_test_split
from utils import utils
import tensorflow as tf
from keras import layers
import random
import sys  # 导入sys模块
# sys.setrecursionlimit(30000)
from sklearn.preprocessing import normalize
MMScaler = MinMaxScaler()
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
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
# 实例化
class VAE(keras.Model):
    def __init__(self, encoder, decoder,data1,data2, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.data1=data1
        self.data2 = data2
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self,data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            print(reconstruction)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mse(data, reconstruction)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class VAEDimensionalAugmentation(object):
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
                data4.append(self.x_train[rm])

                ylabel.append(numPoly)

            #data=np.array(data,dtype=np.float32)
            #print(data.shape)

        ylabel=np.array(ylabel)

        data=np.array(data,dtype=float)
        data2=np.array(data2,dtype=float)
        data3=np.array(data3,dtype=float)
        data4 = np.array(data4, dtype=float)
        input1 = Input(shape=(1761,))
        input2 = Input(shape=(1761,))
        input3 = Input(shape=(1761,))
        encoded1 = Dense(1761, activation='relu')(input1)
        encoded2 = Dense(1761, activation='relu')(input2)
        encoded3 = Dense(1761, activation='relu')(input3)
        d0 = add([encoded1, encoded2])
        encoded = Dense(1024, activation='relu')(d0)

        encoded = Dense(1024, activation='relu')(encoded1)
        z_mean = Dense(1024, name="z_mean")(encoded)
        z_log_var = Dense(1024, name="z_log_var")(encoded)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(input1, [z_mean, z_log_var, z], name="encoder")
        input4 = Input(shape=(1024,))
        encoded = Dense(1024, activation='relu')(input4)
        encoded = Dense(1024, activation='relu')(encoded)
        decoded = Dense(1761, activation='tanh')(encoded)
        decoder = Model(input4, decoded, name='decoder')
        vae=VAE(encoder,decoder,[data2,data3],data4)
        vae.compile(
            optimizer='adam',
            loss='mse'
        )
        dataforvae=[]
        for i in range(len(data2)):
            dataforvae.append([data2[i],data3[i]])
        dataforvae=[[data2,data3],data2]
        print(dataforvae[0])
        vae.fit(data2,
                        epochs = 5,
                        batch_size = 64,
                        shuffle = True)
        _,t,input=encoder.predict(data2)
        intensity0=vae.decoder.predict(input)
        ## for tanh

        return intensity0,ylabel