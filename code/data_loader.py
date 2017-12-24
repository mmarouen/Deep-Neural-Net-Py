#Datasets:
# 1.	Prostate data
# 2.  Vowel (also present in cmu benchmark)
# 3.	Cleveland Heart Disease (not included in the book but added to list)
# 4.	SA Heart
# 5.	Phoneme (aa & ao)
# 6.  Phoneme (all)
# 7.	Bone Mineral Density
# 8.	LA ozone
# 9. Spam
# 10. Protein flow-sytometry
# 11. SVM benchmark1: Astroparticule

import pandas as pd
import sys
sys.path.insert(0,'D:/Spyder/toolkit/')
from preprocess import scaleData

######data1:Prostate data
df=pd.read_table("D:/RProject/DataRepository/ESL/ProstateData.txt")
train=df[df.train=='T'];test=df[df.train=="F"]
train.drop('train',axis=1,inplace=True);test.drop('train',axis=1,inplace=True)
yTrain=train.lpsa;yTest=test.lpsa;
train.drop('lpsa',axis=1,inplace=True);test.drop('lpsa',axis=1,inplace=True)
out0=scaleData(train,resp = yTrain,shuffle = True)
train=out0['X'];yTrain=out0['Y']

######data2: vowel data (results=52% test_tanh_30H_r=0.1_wd=F)
vowelTr=pd.read_table("D:/RProject/DataRepository/ESL/vowel.train",sep=',')
vowelTr.drop('row.names',axis=1,inplace=True)
yTrain=vowelTr.y;train=vowelTr.drop('y',axis=1)
vowelTe=pd.read_table("D:/RProject/DataRepository/ESL/vowel.test",sep=',')
vowelTe.drop('row.names',axis=1,inplace=True)
yTest=vowelTe.y;test=vowelTe.drop('y',axis=1)
out0=scaleData(train,yTrain,shuffle = True);train=out0['X'];yTrain=out0['Y']

#######data3: cleveland heart disease statlog (target 65.6%)
###dataset unclean (response has 3 classes, should be 2)
clevHeartStat=pd.read_csv("D:/RProject/DataRepository/physikaUMKBenchmark/clevelandHeart_statlog.data")
clevHeartStat.columns=["age","sex","cp","restBlPress","chol","fastingSugar","cardioRes","heartRate","exerciceAng","exDepr/rest","exSlope","#majVessels","thal","disease"]
clevHeartStat=clevHeartStat[(clevHeartStat.iloc[:,12] !="?") & (clevHeartStat.iloc[:,11] !="?")]
train=clevHeartStat.iloc[:,:-1];yTrain=clevHeartStat.disease
train.drop('thal',axis=1,inplace=True)

#######data4: SA heart disease
SA=pd.read_table("D:/RProject/DataRepository/ESL/SAheart.data",sep=',')
train=SA.drop(SA.columns[[0,10]],axis=1);yTrain=SA.chd### result=76.57%
out0=scaleData(train,yTrain,shuffle = True);train=out0['X'];yTrain=out0['Y']
SA.famhist[SA.famhist=='Present']=1
SA.famhist[SA.famhist=='Absent']=0
SA.famhist=pd.to_numeric(SA.famhist)
test=train[1:100];yTest=yTrain[1:100];train=train[101:462];yTrain=yTrain[101:462]

######data5: Phoneme (working on 2 phoneme aa & ao)
phoneme=pd.read_table("D:/RProject/DataRepository/ESL/phoneme.data",sep=',')
train=phoneme[phoneme.g.isin(["aa","ao"]) & phoneme.speaker.str.contains('^train')]
train=train.iloc[:,1:257]
test=phoneme[phoneme.g.isin(["aa","ao"]) & phoneme.speaker.str.contains('^test')]
test=test.iloc[:,1:257]
yTrain=phoneme.g[phoneme.g.isin(["aa","ao"]) & phoneme.speaker.str.contains('^train')]
yTest=phoneme.g[phoneme.g.isin(["aa","ao"]) & phoneme.speaker.str.contains('^test')]
yTrain[yTrain=="aa"]=0;yTrain[yTrain=="ao"]=1
yTest[yTest=="aa"]=0;yTest[yTest=="ao"]=1
out0=scaleData(train,yTrain,shuffle = True);train=out0['X'];yTrain=out0['Y'].squeeze()

######data6: Phoneme (working on all 5 phonemes)
phoneme=pd.read_table("D:/RProject/DataRepository/ESL/phoneme.data",sep=',')
train=phoneme[phoneme.speaker.str.contains('^train')];train=train.iloc[:,1:]
test=phoneme[phoneme.speaker.str.contains('^test')];test=test.iloc[:,1:]
yTrain=phoneme.g[phoneme.speaker.str.contains('^train')].astype('category').cat.codes
yTest=phoneme.g[phoneme.speaker.str.contains('^test')].astype('category').cat.codes
out0=scaleData(train,yTrain,shuffle = True);train=out0['X'];yTrain=out0['Y']

######data7:Bone Mineral Density
bone=pd.read_table("D:/RProject/DataRepository/ESL/bone.data")
train=bone.iloc[:,:-1];yTrain=bone.spnbmd
out0=scaleData(train,yTrain,shuffle = True);train=out0['X'];yTrain=out0['Y']

######data8:LA Ozone
laozone=pd.read_table("D:/RProject/DataRepository/ESL/LAozone.data",sep=',')
train=laozone.loc[:,:-1];yTrain=laozone.ozone;
out0=scaleData(train,yTrain,shuffle = True);train=out0['X'];yTrain=out0['Y']

######data9:Spam
spam=pd.read_table("D:/RProject/DataRepository/ESL/spam.data",sep=' ',header=None)
spam1=pd.read_table("D:/RProject/DataRepository/ESL/spam.traintest",header=None).squeeze()
train=spam.loc[spam1==0,0:56];yTrain=spam.loc[spam1==0,57]
test=spam.loc[spam1==1,0:56];yTest=spam.loc[spam1==1,57]
out0=scaleData(train,yTrain,test,yTest,shuffle = True)
train=out0['X'];yTrain=out0['Y'];test=out0['Xtest'];yTest=out0['Ytest']

######data10:Protein flow-cytometry
train=pd.read_table("D:/RProject/DataRepository/ESL/protein.data",header=None,sep=' ',
                      names=["praf","pmec","plcg","PIP2","PIP3","p44/42","pakts473","pKA","pKC","p38","pjnk"])

######data12: SVM benchmark: astroparticule
dat=pd.read_table("D:/RProject/DataRepository/usps",sep=' ',header=None).iloc[:,:-1]
dat2=pd.read_table("D:/RProject/DataRepository/usps.t",sep=' ',header=None).iloc[:,:-1]
yTrain=dat.loc[:,0];yTest=dat2.loc[:,0]
yTrain[yTrain<5]=0;yTest[yTest<5]=0
yTrain[yTrain>4]=1;yTest[yTest>4]=1
train=dat.iloc[:,1:];test=dat2.iloc[:,1:]
import re
train=train.apply(lambda x: x.str.replace(re.compile(".*:"),"").astype(float),axis=1)
test=test.apply(lambda x: x.str.replace(re.compile(".*:"),"").astype(float),axis=1)
out0=scaleData(train,yTrain,test,yTest,shuffle = True)
train=out0['X'];yTrain=out0['Y'];test=out0['Xtest'];yTest=out0['Ytest']

