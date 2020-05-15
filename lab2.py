import pandas as pd
import numpy as np
from scipy import stats


def corr(df,size=10):
    #вычисление кореляционной матрицы с помощью pandas
    corr = df.corr()
    print('\nКореляційна матриця: \n',corr)

def test(df):

    print('\nНормальний тест: \n')
    for i in range(len(df.columns)):
        a, b = stats.normaltest(df[i].values)
        
        if b < 0.05:  # 0.05 is alpha
            print('Для стовпця ', i,' прийнято')
        else:
            print('Для стовпця ', i,' відхилено')
    

def dataset (a):
    #считывание датасета и его нумерация
    dataframe = a.loc[:, a.columns != 0]
    dataframe.columns = [i for i in range(6)]
    return dataframe;

def column_info(dataframe):
    #вывод статистических значений по каждому столбцу
    #датасета, а так же их сохранение для дальнейших вычислений
    names = ['Середнє', 'Стандартне відхилення']
    mean, deviation = [], []
    for i in range(len(dataframe.columns)):
        mean.append(dataframe[i].values.mean())
        deviation.append(dataframe[i].values.std())
    zipped = list(zip(names, [mean, deviation]))
    data = dict(zipped)
    stats = pd.DataFrame(data)
    print(stats)
    return mean, deviation

def normalization(dataset, mean, std):
    #функция нормализирует датасет по формуле z = (x-mean)/deviation
    names = dataset.columns.values
    dz_cols = [(dataset[i].values-mean[i])/std[i] for i in range(len(dataset.columns))]
    zipped = list(zip(names, dz_cols))
    data = dict(zipped)
    load = pd.DataFrame(data)
    print('\nНормалізований датасет\n')
    print(load)
    return load



#main
a = pd.read_csv('kpi5.txt', sep='\s+', header=None)
dataframe = dataset(a)
mean, deviation = column_info(dataframe)
normalized = normalization(dataframe,mean,deviation)
corr(dataframe)
test(dataframe)




