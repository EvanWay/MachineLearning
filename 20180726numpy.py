import pandas as pd
df_train=pd.read_csv("C:\Users\Evan\Desktop\learing\wc2018players.csv")  #DataFrame
df_test=pd.read_csv("C:\Users\Evan\Desktop\learing\wc2018players.csv")

df_test_negative=df_train.loc[df_train['Type']==0][['Clump Thickness',  'Cell Size' ]]

df_test_positive=df_train.loc[df_train['Type']==1][['Clump Thickness',  'Cell Size' ]]


import matplotlib.pyplot as plt

#print(df_train)
plt.subplot(231)
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=10,c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=10,c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')

plt.subplot(232)
import numpy as np
intercept=np.random.random([1])
coef=np.random.random([2])
lx=np.arange(0,12)
ly=(-intercept-lx*coef[0])/coef[1]  #(-a-x*b)/c
plt.plot(lx,ly,c='blue')
plt.xlabel('(-%f-x*%f)/%f'%(intercept,coef[0],coef[1]))



plt.subplot(233)
ly=2.0*(-intercept-lx*coef[0])/coef[1]+10  #(-a-x*b)/c
plt.plot(lx,ly,c='blue')
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=10,c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=10,c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')

plt.subplot(234)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(df_train[['Clump Thickness','Cell Size']][:10],df_train['Type'][:10])
#print('accuracy(10):',lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type']))

intercept=lr.intercept_
coef=lr.coef_[0,:]
#print(intercept,coef)
ly=(-intercept-lx*coef[0])/coef[1]
plt.plot(lx,ly,c='green')
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=10,c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=10,c='black')
plt.xlabel('Clump Thickness-accuracy(10):'+str(lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type'])))
plt.ylabel('Cell Size')



plt.subplot(235)
lr=LogisticRegression()
lr.fit(df_train[['Clump Thickness','Cell Size']][:200],df_train['Type'][:200])
#print('accuracy(200):',lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type']))
plt.xlabel('Clump Thickness-accuracy(200):'+str(lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type'])))
ly=(-intercept-lx*coef[0])/coef[1]
plt.plot(lx,ly,c='green')
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=10,c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=10,c='black')


plt.subplot(236)
lr=LogisticRegression()
lr.fit(df_train[['Clump Thickness','Cell Size']],df_train['Type'])
#print('accuracy(all):',lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type']))
plt.xlabel('Clump Thickness-accuracy(all):'+str(lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type'])))
ly=(-intercept-lx*coef[0])/coef[1]
plt.plot(lx,ly,c='green')
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=10,c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=10,c='black')
plt.show()
