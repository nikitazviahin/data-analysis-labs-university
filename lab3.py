import pandas as pd
import numpy 
from numpy.linalg import lstsq
from scipy import stats
import matplotlib.pyplot as plt
import seaborn
from random import choice, sample

'''Обробка датасету для подальшої роботи'''
initialset = pd.read_csv('kpi5.txt', sep='\s+', header=None)
dataset = initialset.loc[:, initialset.columns != 0]
dataset.columns = [i for i in range(6)]
print(dataset)


'''Обчислення залежної змінної'''
numpy.random.seed(1)
k = 5
rand = choice(numpy.random.uniform(0, 1, 1000))
a1 = k + rand
a2 = -k/3 + rand
a3 = k/2 - rand
y = a1*dataset[0] + a2*dataset[2] + a3*dataset[4] + 2*(rand - 0.5)
dataset.insert(6, 6, y, True)


'''Метод наймнеьших квадратів
   для знаходження лінійної функціональної залежності'''
y = dataset[0].values
x = dataset[1].values
summay = numpy.sum(y)
n = 21600
summax = numpy.sum(x)
summax_sq = numpy.sum(x ** 2)
xsummay = numpy.sum(x * y)
b = (n*xsummay - summay*summax)/(n*summax_sq - summax*summax)
a = (summay - b*summax)/n


'''Графічне представлення'''
plt.style.use('seaborn')
z = numpy.linspace(-2, 2, 100)
plt.scatter(x, y, color='#00008B', alpha=0.90, marker='.')
plt.plot(z, b*z + a, color='#FF0000', alpha=0.95,  linewidth=3)
plt.title('Графік регрессії')
plt.xlim(-1, 2)
plt.ylim(-5, 5)
plt.show()

'''Підрахування кількості викидів
   та їх видалення'''
R_prescott = 4
vybrosi = 0
bursts = True
def func(x, a, b): return a + x * b
while (bursts is True):

    y_hat = numpy.array([func(j, a, b) for j in x])
    ei = abs(y - y_hat)
    max_ei = max(ei)
    ei_sq = ei / numpy.sqrt(numpy.sum(ei**2))
    max_ei_sq = max(ei_sq)
    indx_max = numpy.where(ei_sq == max_ei_sq)[0][0]
    R = numpy.sqrt(len(ei)) * max_ei_sq
    if R > R_prescott:
        indeces = [i for i in range(len(ei)) if ei[i] == ei[indx_max]]
        vybrosi += len(indeces)
        for i in indeces:
            y = numpy.delete(y, i)
            x = numpy.delete(x, i)
            summay = numpy.sum(y)
            n = len(y)
            summax = numpy.sum(x)
            summax_sq = numpy.sum(x ** 2)
            xsummay = numpy.sum(x * y)
            b = (n*xsummay - summay*summax)/(n*summax_sq - summax*summax)
            a = (summay - b*summax)/n
    else:
        bursts = False
print('Кількість викидів:', vybrosi)

'''Графічне представлення без викідів'''
plt.style.use('seaborn')
z = numpy.linspace(-2, 2, 100)
plt.scatter(x, y, color='#00008B', alpha=0.90, marker='.')
plt.plot(z, b*z + a, color='#FF0000', alpha=0.95,  linewidth=3)
plt.title('Графік регрессії без викидів')
plt.xlim(-1, 2)
plt.ylim(-5, 5)
plt.show()

'''Перевірка критерію'''
t_student = 1.96
x_mean = numpy.mean(x)
lnght = len(x)
s_squared = 1/(lnght - 2) * numpy.sum((y - a - b*x)**2)
sx_squared = 1/(lnght - 1) * numpy.sum((x - x_mean)**2)
s_beta = numpy.sqrt(s_squared) / (numpy.sqrt(sx_squared * (lnght - 1)))
print('Abs(b) = {}'.format(abs(b)))
print('t(Student) * S(beta) = {}'.format(t_student*s_beta))
if abs(b) > t_student * s_beta:
    print("'b' значущій!")
else:
    print("'b' не значущій!")
s_alpha = numpy.sqrt(s_squared*(1/lnght + x_mean**2/((lnght-1)*sx_squared)))
print('\nAbs(a) = {}'.format(abs(a)))
print('t(Student) * S(alpha) = {}'.format(t_student*s_alpha))
if abs(a) > t_student * s_alpha:
    print("'a' значущій!")
else:
    print("'a' не значущій!")


X = []
X.append([1 for i in range(21600)])
for i in range(6):
    X.append(dataset[i].values)
XX = numpy.transpose(X)
Y = [[i] for i in dataset[6].values]
aa = numpy.around(lstsq(numpy.transpose(X), Y, rcond=None)[0], decimals=4)
coefs = [aa[i][0] for i in range(len(aa))]
Q = round(numpy.dot(numpy.transpose(Y - XX.dot(aa)), (Y - XX.dot(aa)))[0][0], 10)
Qr = round(numpy.dot(numpy.transpose(XX.dot(aa)), (XX.dot(aa)))[0][0], 10)
F = (Qr/(len(dataset.columns)))/(Q/(len(dataset[0].values) - 5))
Fcr = 2
print("test")
if F > Fcr:
    print('\nГіпотеза H0 відхилена')
else:
    print('\nГіпотеза H0 прийнята')
print('\nРегресійна модель')
print('y = {} + {}*х1 + {}*х2 + {}*х3 + {}*х4 + {}*х5 + {}*х6'.format(coefs[0],
                                                                          coefs[1],
                                                                          coefs[2],
                                                                          coefs[3],
                                                                          coefs[4],
                                                                          coefs[5],
                                                                          coefs[6]))
