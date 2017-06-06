import numpy as np
import csv

class futuresData:
    def __init__(self):
        self.mFuturesNum = -1
        self.mInforFieldsNum = -1
        self.mLength = -1
        self.mPoundage = -1
        self.mData = []
        self.mDate = []
        self.mPrice = []

    def loadData_moreday0607(self):
        with open('data/moreday0607.csv') as f:
            print('[A3C_data]Loading data from data/moreday0607.csv ...')
            self.mFuturesNum = 6
            self.mInforFieldsNum = 10
            self.mLength = 0
            self.mPoundage = 0
            reader = csv.reader(f)
            i = 0
            for row in reader:
                if i < 4:
                    i += 1
                    continue
                else:
                    baddata = False
                    idata = np.zeros([self.mFuturesNum,self.mInforFieldsNum])
                    iprice = np.zeros(self.mFuturesNum)
                    for j in range(0,self.mFuturesNum):
                        dateidx = j * (self.mInforFieldsNum + 2)
                        for k in range(0,self.mInforFieldsNum):
                            istring = row[dateidx + k + 1]
                            if len(istring) == 0 :
                                baddata = True
                                break
                            idata[j][k] = float(istring)
                        if baddata == True:
                            break
                        iprice[j] = idata[j][1]
                    if baddata == True:
                        i += 1
                        continue
                    self.mData.append(idata.reshape(self.mFuturesNum * self.mInforFieldsNum))
                    self.mDate.append(row[0])
                    self.mPrice.append(iprice)
                    i += 1
                    self.mLength += 1
                    if i >= 50:
                        break;
        print('[A3C_data]Successfully loaded ' + str(self.mLength) + ' data')

    def getObservation(self,time):
        return self.mData[time]

    def getPrice(self,time):
        return self.mPrice[time]
