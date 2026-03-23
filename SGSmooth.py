import numpy as np
from scipy.signal import savgol_filter
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
import os
# from als import als_baseline
class SGsmooth():

    def __init__(self,fileName):
        self.fileName=fileName
        self.data=self.readFile(self.fileName)

    def readFile(self,fileName):
        if os.path.exists(fileName):
            print('1')
    def parseData(self,fileName,begin,end):
        dataSet = pd.read_csv(fileName, header=None, engine='python', encoding='latin-1')
        data = dataSet.iloc[:326, :]
        waveNumber = data.iloc[0, begin:end]
        waveNumber = np.array(waveNumber)
        intensity = data.iloc[1:, begin:end]
        intensity = np.array(intensity)
        polymerName=data.iloc[1:,1866]
        polymerName=np.array(polymerName)
        return waveNumber,intensity,polymerName
    def np_move_avg(a, n, mode="same"):
        return (np.convolve(a, np.ones((n,)) / n, mode=mode))
if __name__ == '__main__':

    Sg=SGsmooth('216_2018_1156_MOESM2_ESM.csv')
    waveNumber,intensity,polymerName=Sg.parseData('216_2018_1156_MOESM2_ESM.csv',1,1864)

    # 可视化图线



  #   from scipy.interpolate import make_interp_spline
  #
  #
  #   y = intensity[0]
  #   x_smooth = np.linspace(waveNumber.min(), waveNumber.max(), 1000)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
  #
  #   y_smooth = make_interp_spline(waveNumber, y)(x_smooth)
  # #  plt.plot(x, y_smooth)
  #   plt.show()
    from scipy.interpolate import make_interp_spline

    fig, ax = plt.subplots()
    x=np.sort(waveNumber)
    print(x)
    yNoise=np.random.rand(intensity.shape[1]) * 0.008
    x_smooth = np.linspace(x.min(), x.max(), 80)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
    y_smooth = make_interp_spline(x, intensity[5])(x_smooth)
    y = savgol_filter(intensity[5]+yNoise, 65, 2, mode='nearest')
    ax.plot(waveNumber, intensity[5]+yNoise, 'r', label='original')

    #ax.plot(waveNumber, y, 'r', label='savgol', linestyle='solid')
    #ax.plot(x_smooth[::-1], y_smooth, 'g', label='inter', linestyle='dashdot')
    # y_alsBaseline=als_baseline(y)
    ax.plot(waveNumber,y,'b',label='SG_Smooth',linestyle='dashed')
    # x = np.linspace(0, 2, 100)
    # fig, ax = plt.subplots()
    # ax.plot(x, x, label='linear')
    # ax.plot(x, x ** 2, label='quadratic')
    # ax.plot(x, x ** 3, label='cubic')

    # plt.show()

    #plt.plot(np.convolve(intensity(5),np.ones((50,)),mode='full'),'y',linestyle='dashed')
    # for i in range(1,80):
    #     yconv=np.convolve(intensity[5], np.ones((i,)) / 50,mode='same')
    #     if max(yconv)>=max(intensity[5]):
    #         ax.plot(waveNumber, yconv, 'grey', label='cov', linestyle='solid')
    #         print(i)
    #         break
    # modes = ['full', 'same', 'valid']
    #yconv = np.convolve(intensity[5], np.ones((63,)) / 50, mode='same')
    #ax.plot(waveNumber, yconv, 'grey', label='cov', linestyle='solid')
    # for m in modes:
    #     plt.plot(np_move_avg(np.ones((200,)), 50, mode=m))
    #
    # plt.axis([-10, 251, -.1, 1.1])
    #
    # plt.legend(modes, loc='lower center')
    # sampling_step=waveNumber[1]-waveNumber[0]
    # deriv1_method1 = signal.savgol_filter(intensity[10], 11, 2, deriv=2, delta=sampling_step,mode='nearest')
    #
    # #derivate the spectrum intensity
    # ax.plot(waveNumber, deriv1_method1, 'purple', label='SGderiv1', linestyle='dashdot')
    # dy = np.diff(signal.savgol_filter(intensity[10], 11, 2,mode='nearest'))
    # dx = np.diff(waveNumber)
    #
    # deriv1_method2 = dy / dx
    # ax.plot(waveNumber[1:], deriv1_method2, 'b', label='SGderiv2', linestyle='dashdot')
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('normal') for label in labels]
    [label.set_fontstyle('normal') for label in labels]
    ax.set_xlabel('waveLength',font2)
    ax.set_ylabel('intensity' ,font2)
    ax.set_title('SG_Smooth',font2)
    ax.legend()
    plt.show()
