import numpy

from utils import  utils
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import pipeline
def normalizedWavelength(wavenumbers):
    if wavenumbers[0] > wavenumbers[-1]:
        rng = wavenumbers[0] - wavenumbers[-1]
    else:
        rng = wavenumbers[-1] - wavenumbers[0]
    half_rng = rng / 2
    normalized_wns = (wavenumbers - np.mean(wavenumbers)) / half_rng
    return normalized_wns
def cal_deriv(x, y):  # x, y的类型均为列表
    diff_x = []  # 用来存储x列表中的两数之差
    for i, j in zip(x[0::], x[1::]):
        diff_x.append(j - i)

    diff_y = []  # 用来存储y列表中的两数之差
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)

    slopes = []  # 用来存储斜率
    for i in range(len(diff_y)):
        slopes.append(diff_y[i] / diff_x[i])

    deriv = []  # 用来存储一阶导数
    for i, j in zip(slopes[0::], slopes[1::]):
        deriv.append((0.5 * (i + j)))  # 根据离散点导数的定义，计算并存储结果
    deriv.insert(0, slopes[0])  # (左)端点的导数即为与其最近点的斜率
    deriv.append(slopes[-1])  # (右)端点的导数即为与其最近点的斜率

    return deriv
from sklearn import  linear_model

def cal_2nd_deriv(x,y):
    return cal_deriv(x, cal_deriv(x, y))
def cal_3rd_deriv(x,y):
    return cal_deriv(x, cal_2nd_deriv(x, y))
polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData11('D4_4_publication11.csv', 2, 1763)
# from scipy.signal import savgol_filter
#
# def moving_average(interval, windowsize):
#     window = np.ones(int(windowsize)) / float(windowsize)
#     re = np.convolve(interval, window, 'same')
#     return re
# ita1=savgol_filter(intensity[0], 43, 2, mode='nearest')
# ita1=np.array(ita1)
# print(ita1.shape)
# defy=cal_deriv(waveLength,ita1)
# poly_reg=PolynomialFeatures(degree=1000)
# waveLength=np.array(waveLength)
# waveLength=normalizedWavelength(waveLength)
#
# linreg= linear_model.LinearRegression()
#
#
# defy2=cal_2nd_deriv(waveLength,ita1)
# defy3=cal_3rd_deriv(waveLength,ita1)
# peak1=[]
#
# import scipy.signal as sg
# peaks=sg.find_peaks(ita1)[0]
# # print(len(peaks))
# # for i in range(len(peaks)):
# #     peak1.append(peaks[i])
# #     print(peaks[i])
# # intensity1=np.array(intensity[0])
# # print(peak1)
# # # intensity1.reshape(1761)
# # print(intensity1[peak1])
# # max=sg.argrelmax(intensity[0])
# # min=sg.argrelmin(intensity[0])
# # print(peaks)
# # print(max)
# # print(min)
# # model=pipeline.make_pipeline(
# #
# #     PolynomialFeatures(20),
# #
# # )
#
# intensityforPoly=numpy.array(intensity[0])
# intensityforPoly=intensityforPoly.reshape(-1,1)
#
#
# waveLength=waveLength.reshape(-1,1)
#
#
# test_x=np.linspace(waveLength.max(),waveLength.min(),1761)
# x_poly=poly_reg.fit_transform(waveLength)
# linreg.fit(x_poly,intensityforPoly)
#
#
# print(test_x)
# test_x=test_x.reshape(-1,1)
# test_x=poly_reg.fit_transform(test_x)
# pre=linreg.predict(test_x)
# defy=np.array(defy)
# defy2=np.array(defy2)
# defy3=np.array(defy3)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=4, ncols=1)
# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 30,
#          }
# ita=np.array(intensity[0])
#
# print(ita.shape)
# for i in range(1761):
#
#     if i<=800:
#         if defy2[i]<0 and defy2[i+1]>0:
#             ita[i]+=np.random.randint(10,11)
#     if 800<i<1760:
#         if defy2[i]<0 and defy2[i+1]>0:
#             ita[i]+=np.random.randint(10,11)*0.5
# from scipy.signal import savgol_filter
# def moving_average(interval, windowsize):
#     window = np.ones(int(windowsize)) / float(windowsize)
#     re = np.convolve(interval, window, 'same')
#     return re
#
# ita= savgol_filter(ita, 43, 2, mode='nearest')
# ita=moving_average(ita,5)
# ax[0].plot(waveLength, intensity[0], '-k')
# ax[1].plot(waveLength, ita, '-k')
# ax[2].plot(waveLength, pre, '-b')
#
# ax[3].plot(waveLength, defy3*100, '-r')
# plt.show()