"""
多项式回归
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp
import sklearn.metrics as sm
import sklearn.preprocessing as sp
import sklearn.pipeline as pl
from utils import utils
from FTIR_PolynomialRegression import normalizedWavelength
polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData11('D4_4_publication11.csv', 2, 1763)
# 把x改为n行1列  这样才可以作为输入交给模型训练
waveLength=np.array(waveLength)
waveLength=waveLength.reshape(-1,1)
# 训练多项式回归模型
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#
waveLength=np.array(waveLength,dtype=float)
waveLength=normalizedWavelength(waveLength)
waveLength=waveLength.reshape(1761,)

x_test = np.linspace(waveLength.max(),waveLength.min(),1000)
print(waveLength.shape)

z1 = np.polyfit(waveLength, intensity[0], 60) # 用7次多项式拟合，可改变多项式阶数；
p1 = np.poly1d(z1)
plt.scatter(waveLength,intensity[0])
plt.plot(waveLength,p1(waveLength))
plt.show()


# 评估回归模型的误差
# # 平均绝对值误差  1/m∑|预测输出-真实输出|
# print(sm.mean_absolute_error(y, pred_y))
# # 平均平方误差  sqrt(1/m∑(预测输出-真实输出)^2)
# print(sm.mean_squared_error(y, pred_y))
# # 中位数绝对值误差  median(|预测输出-真实输出|)
# print(sm.median_absolute_error(y, pred_y))
# # r2得分 (0,1]的一个分值,分数越高,误差越小
# print(sm.r2_score(y, pred_y))

# mp.figure('Linear Regression', facecolor='lightgray')
# mp.title('Linear Regression', fontsize=18)
# mp.xlabel('X', fontsize=16)
# mp.ylabel('Y', fontsize=16)
# mp.tick_params(labelsize=12)
# mp.grid(linestyle=':')
# mp.scatter(waveLength, intensityforloop[0], s=60, c='dodgerblue',
# 	label='Points')
# mp.plot(test_x, pred_test_y, c='orangered',
# 	linewidth=2, label='Regression Line')
# mp.legend()
# mp.show()
