
import pandas as pd
import numpy as np
import os
class calculateReportScores:
    def __init__(self):
        pass

    def readFromFile(self,fileName):
        fileList=[]
        scoreForReport = np.zeros((3,12))
        #scoreForReport = np.zeros((3, 5))
        count = len(os.listdir(fileName))
        for item in os.listdir(fileName):  # 文件、文件夹名字
            print(item)
            fileList.append(item)

        for item in fileList:
            data = pd.read_csv(fileName + '/' + item, header=None,
                               encoding='latin-1',
                               keep_default_na=False,
                               low_memory=False)
            data = data.iloc[1:4, 1:13]
            #data = data.iloc[1:4, 1:6]
            data = np.array(data,dtype=np.float32)
            scoreForReport=scoreForReport+data



        return scoreForReport/count
if __name__ == '__main__':
    utils=calculateReportScores()
    scoreForReport=data=utils.readFromFile('SVCAE_plus_report')
    print(scoreForReport)
    statisticForMetrics=pd.DataFrame(scoreForReport)
    statisticForMetrics.to_csv('report/SVCAE_plus_report.csv')
    # for _ in os.listdir('SVC_PCA_report'):  # 文件、文件夹名字
    #     print(_)
    # count = len(os.listdir('SVC_PCA_report'))
    # print(count)