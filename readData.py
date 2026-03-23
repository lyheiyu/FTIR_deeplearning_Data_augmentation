import numpy
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
class readFile:
    def __init__(self):
        pass
    def readData(self,folderName):
       fileNameList=os.listdir(folderName)
       print(folderName)
       count=len(os.listdir(folderName))
       intensityList=[]
       fig, ax = plt.subplots()
       plt.tick_params(labelsize=20)
       plt.xlim(4000, 500)
       for i in range(count):
           data = pd.read_csv(folderName + '/' + fileNameList[i], header=None,
                              encoding='latin-1',
                              keep_default_na=False,
                              low_memory=False)
           waveLength=data.iloc[:,0]
           intensity:np.array()=data.iloc[:,1]
           print(intensity.shape)
           intensityList.append(intensity)
           ax.plot(waveLength, intensity)
           # intensityList.append(intensity)

       plt.xlabel('Wavelength', size=30)
       plt.ylabel('Intensity', size=30)
       plt.title(folderName, size=30)
       plt.show()
       # intensityList=numpy.array(intensityList)

       return intensityList,waveLength

if __name__ == '__main__':
    rf=readFile()
    # foldList=['PE','PET','PMMA','PP','PS','PVC']
    foldList = ['PVC']
    for item in foldList:
        print(item)
        rf.readData(item)