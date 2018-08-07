import pandas as pd
import numpy as np
pd.read_csv('D:/python/2018File/20180726/my.csv')
print('-----------------------')

#序列行转为列 index 从 0-n
s=pd.Series([1,2,3,4,5,np.nan,4,6]);
#print(s)

#二维数组日期开始，递增几个 
dates=pd.date_range('2018-01-01',periods=6)
#print(dates)

# 6行 4列， index 用上面的DATEs , 列用的是A B C D
#df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
#print(df)

 # 默认INDEx 是0 -N
df=pd.DataFrame(np.random.randn(6,4),columns=list('ABCD'))
#print(df)

# 一行5列，且第一行的index 值为5 ，列名分别是 'col1','col2','col3','col4','col5'
df=pd.DataFrame([[1,2,3,4,5]],index=[5],columns=['col1','col2','col3','col4','col5'])
#print(df)

# 一列 5行，index 值自己设置，列名为coll
df=pd.DataFrame([1,2,3,4,5],index=['r1','r2','r3','r4','r5'],columns=['col1'])
#print(df)

 
#字典二维表，缺少补值
df2=pd.DataFrame({'A':1,
                  'B':pd.Timestamp('2018-01-02'),
                  'C':pd.Series(1,index=list(range(4)),dtype='float32'),
                  'D':np.array([3]*4,dtype='int32'),
                  'E':pd.Categorical(['test','train','test','train']),
                  'F':'foo'
                  })
#print(df2)
print('-----------------------下-------------------------------------')
#print(df2.dtypes)
#print(df2.any())

# 前两行
#print(df2.head(2))
# 后两行
#print(df2.tail(2))

# 打印索引列
#print(df2.index)
#打印列明
#print(df2.columns)
#打印值
#print(df2.values)

#print(df2.describe())

#print(df2.T)
#print(df2.sort_index(axis=1,ascending=False))
#print(df2.sort_values(by='E',ascending=False))
#print(df2['A'].sort_values())
#print(df2[['A','B']].sort_values(by='A'))

#//行切片
print(df2[0:2])
print(df2[2:3])
print('-----------------------df3-------------------------------------')
df3=pd.read_csv('D:/python/2018File/20180726/my.csv','\t')
ds=df3.describe()
 
#50% 行索引到 max 行索引，说明index 可以为object ,其索引也是
#print(ds['50%':'max']) 
#第一列
#print(df3.loc[1])

print(df3.loc[0:10,['Number1','Type','Value']])
print('-----------------------df3-------------------------------------')
#print(df3.loc[3:4,['Number1','Type','Value']])
#print(df3.loc[3,'Number1'])
#print(df3.at[3,'Number1'])
#print(df3.iloc[3])
#print(df3.loc[3])
#print(df3.iloc[2:4,2:4])
#print(df3.iloc[1,2])
#print(df3.iat[1,2])
#print(df3[df3.Number1<10])
#print(df3[df3<3])
df4=df3.copy()
#print(df4)
print('-----------------------df444444444-------------------------------------')
df4['K']=range(600,646)
#print(df4)
#print(df4[df4['K'].isin(range(610,619,3))])
#print(len(df4))

df4['R']=np.random.randn(len(df4))
#print(df4)

#模块的绑定值处理
df4.at[0,'K']=1000
df4.iloc[1,4]=2000
df4.iat[0,1]=8
df4.loc[2,['K']]=2100
df4.loc[3,['K','R']]=[2110,1.0001]
df4.loc[3:5,['K','R']]=[[2111,1.0011],[2112,1.0021],[2113,1.0031]]
df4.iloc[0:3,4:6]=[[1111,2.0011],[1112,2.0021],[1113,2.0031]]
#print(df4)




#print(df4.at[0,'K'],df4.at[1,'K'],df4.at[2,'K'],df4.at[3,'K'],df4.at[3,'R'])
#print(df4.loc[0:5])
print(len(df4))
df4.loc[:,'N']=np.array([np.random.randn(1)]*len(df4))
#print(df4)
print('-----------------------555555555555555555555555555555555555555555-------------------------------------')
df5=pd.DataFrame([[10,20,30],[40,50,60],[70,80,90]],columns=['col1','col2','col3'],index=['A','B','C'])
df5[df5>70]=-df5
#print(df5)
df5=df5.reindex(index=['a','b','c'],columns=list(df5.columns)+['col4'])
df5.loc['b':'c','col4']=1
#print(df5)
#print(df5.dropna(how='any'))
df5=df5.fillna(value=5)
print(df5)
print('-----------------------555555555555555555555555555555555555555555-------------------------------------')
#print(df5.mean())
#print(df5.mean(1))
print('-----------------------555555555555555555555555555555555555555555-------------------------------------')
s=pd.Series([1,3,5,np.nan,6,8],index=pd.date_range('2018-9-1',periods=6));
#print(s.shift(2))
#print(s.shift(-2))
#print(s.sub(s))
#print(df5.sub(s,axis='index'))

#print(df5)
print('-----------------------666666666666666666666666666666666666666666666666666666-------------------------------------')
#print(df5.apply(np.cumsum))
#print(df5.apply(lambda x:x.max()-x.min()))
#print(df5.apply(lambda y:y.max()-y.min()))
print('-----------------------###################77777777777777777777777777777-------------------------------------')

df=pd.DataFrame(np.random.randn(4,3),columns=list('bde'),index=['utah','ohio','texas','oregon'])
print(df)

f=lambda x:x.max()-x.min()
#默认情况下会以列为单位，分别对列应用函数
t1=df.apply(f)
print(t1)
print('-----------------------################### 列-------------------------------------')

t2=df.apply(f,axis=1)
print(t2)
print('-----------------------################### 行-------------------------------------')

#除标量外，传递给apply的函数还可以返回由多个值组成的Series
def f(x):
    return pd.Series([x.min(),x.max()],index=['min','max'])
t3=df.apply(f)
#从运行的结果可以看出，按列调用的顺序，调用函数运行的结果在右边依次追加
print(t3)
print('-----------------------################### 2位的精度 取x的余数 -------------------------------------')
f=lambda x: '%.2f'%x
t3=df.applymap(f)
print(t3)

#注意，这里之所以叫applymap,是因为Series有一个永远元素级函数的map方法
t4=df['e'].map(f)
print(t4)










