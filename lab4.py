import pandas as pd
import numpy as np
from pandas import DataFrame
import seaborn as sn
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy import stats
import random
import statsmodels.api as sm
pd.options.mode.chained_assignment = None
import matplotlib.patches as matplotpatches

variant = 5
n = 21600

df = pd.read_csv('kpi5.txt',sep='\s+', engine='python',header=None,
                   names=["Delta", "x1", "x2", "x3", "x4", "x5", "x6"])
#формуємо x4,x5,x6
i=0
while i<n:
    df['x4'][i] = df['x1'][i]+2*df['x2'][i]+ random.random()  - 1/2
    df['x5'][i] = df['x1'][i]-3*df['x2'][i]+2*(random.random()-1/2)
    df['x6'][i] = 2*df['x2'][i]-df['x3'][i]+3*(random.random()-1/2)
    i+=1

listmx = df.apply(np.mean)
listmx = list(listmx)
listmx = listmx[1:]

plt.style.use('seaborn')
for i in range(len(df.columns)-1):
    plt.subplot(2, 3, i+1)
    plt.hist(df['x{}'.format(i+1)].values, bins=300, ec='blue')
    plt.title('Factor X{}'.format(i+1))
plt.tight_layout()
plt.show()

#Підрахування основних статистик (середнє, дисперсія)
sqm1 = sum(df['x1']**2)/n
sqm2 = sum(df['x2']**2)/n
sqm3 = sum(df['x3']**2)/n
sqm4 = sum(df['x4']**2)/n
sqm5 = sum(df['x5']**2)/n
sqm6 = sum(df['x6']**2)/n

variance_x1 = sqm1 - ((listmx[0])**2)
variance_x2 = sqm2 - ((listmx[1])**2)
variance_x3 = sqm3 - ((listmx[2])**2)
variance_x4 = sqm4 - ((listmx[3])**2)
variance_x5 = sqm5 - ((listmx[4])**2)
variance_x6 = sqm6 - ((listmx[5])**2)

mean_list = []
mean_list.append(sqm1)
mean_list.append(sqm2)
mean_list.append(sqm3)
mean_list.append(sqm4)
mean_list.append(sqm5)
mean_list.append(sqm6)

variance_list = []
variance_list.append(variance_x1)
variance_list.append(variance_x2)
variance_list.append(variance_x3)
variance_list.append(variance_x4)
variance_list.append(variance_x5)
variance_list.append(variance_x6)

print('Основний датасет \n',df)
print('Середні значення по факторам \n')
i = 0
while i < 6:
    print(mean_list[i])
    i+=1
print('Дисперсія по факторам \n')
i = 0
while i < 6:
    print(variance_list[i])
    i+=1
#нормалізація даних
names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
dz = [stats.zscore(df['x{}'.format(i+1)].values, ddof=1) for i in range(len(df.columns) - 1)]
zipp = list(zip(names, dz))
data = dict(zipp)
norm_df = pd.DataFrame(data)
print('\nНормалізований датасет\n', norm_df)
#кореляційна матриця
corrMatrix = norm_df.corr()
print(corrMatrix)
#власні числа та вектори

w, v = LA.eig(corrMatrix)
print('Власні числа та вектори відповідно\n')
print(w, '\n')
print(v)
print('Сумма w ', sum(w))
new_df = norm_df.iloc[:, :3]
L = np.array([v[i] for i in range(3)])
X = np.array([new_df['x{}'.format(i+1)].values for i in range(3)])
Z = X.T @ L
Z = pd.DataFrame(Z[:, :3], columns=['x1', 'x2', 'x3'])
print('Матриця головних компонент\n',Z)
x = [i for i in range(3)]
Z_ = np.array(Z.sum()[x])
random.seed(1)
from sklearn.linear_model import LinearRegression

y_hat = 3*norm_df.x1 - 2*norm_df.x2 + norm_df.x3 - norm_df.x4 - 3*norm_df.x5 + 5*norm_df.x6 + 4*(random.random() - 0.5)
mlr = LinearRegression()
mlr.fit(df, y_hat)
inter = round(mlr.intercept_, 4)
coefs = np.around(mlr.coef_, decimals=4)
print('Модель: Y = {} + {}*X1 + {}*X2 + {}*X3 + {}*X4 + {}*X5 + {}*X6'.format(inter, coefs[0],
                                                                                      coefs[1],
                                                                                      coefs[2],
                                                                                      coefs[3],
                                                                                      coefs[4],
                                                                                      coefs[5]))





