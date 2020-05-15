#Звягин Никита, КМ-73 Лаб. 1-1, Вариант 5

import numpy as np
import pandas as pd
    

def anova(dataset):
    columns_s = []
    q1 = 0
    for i in range(len(dataset.columns)):
        columns_s.append(sum(dataset[i].values))
        q1 += np.square((dataset[i].values)).sum()
        
    q2 = 1/len(dataset[1].values) * sum(np.square(columns_s))
    q3 = 1/(len(dataset[1].values) * len(dataset.columns)) * pow(sum(columns_s), 2)
    
    so_sq = (q1 - q2)/((len(dataset[1].values) - 1)*len(dataset.columns))
    sa_sq = (q2 - q3)/(len(dataset.columns) - 1)
    
    factor = sa_sq / so_sq
    
    f1 = len(dataset.columns) - 1
    f2 = len(dataset.columns)* (len(dataset[1].values) - 1)
    
    fisher_crit_number = 2.21
    print("sa_sq/so_sq = ",factor)
    print("f_aplha = ", fisher_crit_number)
    if factor > fisher_crit_number:
        print('\nПринимаем гипотезу')
    else:
        print('\nОтклоняем гипотезу')


def variance(dataset):
    varc = []
    
    for i in range(len(dataset.columns)):    #считаем дисперсию
        varc.append(dataset[i].values.var()) #для каждого столбца датасета
    print("Дисперсия каждого столбца")
    for i in varc:
        print(i)
    print("\nКритерий сравнения g: ")
    g = max(varc) / sum(varc)           #критерий сравнения
    print(g) 


a = pd.read_csv('kpi5.txt', sep='\s+', header=None) 
dataset = a.loc[:, a.columns != 0]        #считывание датасета
dataset.columns = [i for i in range(6)]   #нумерация столбцов
variance(dataset)
anova(dataset)
    
    
