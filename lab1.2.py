import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns



def prep(df):     #обробка датасету для отримання середніх значень групі 
    sl = int(len(df[1].values)/4) #яка відноситься до фактору b
    dff = []
    for i in range (len(df.columns)):
        dff.append(df[i].values)
    b1 = []
    b2 = []
    b3 = []
    b4 = []
    dffn = [b1, b2, b3, b4]
    for j in range (len(df.columns)):
        b1.append(dff[j][:sl])
        b2.append(dff[j][sl:(2*sl)])
        b3.append(dff[j][(2*sl):(3 * sl)])
        b4.append(dff[j][(3*sl):(4 * sl)])
    a = []
    for i in range(len(dffn)):
        b = [np.mean(dffn[i][j]) for j in range(len(dffn[i]))]
        a.append(b)
    names = [0, 1, 2, 3]
    zipped = list(zip(names, a))
    data = dict(zipped)
    stats_t = pd.DataFrame(data)
    stats = stats_t.transpose()
    return stats

def twoway_anova(df):
    sumsq1 = 0
    for i in range(6):
        for j in range(4):
            sumsq1 += df[i][j]**2
    print('Q1 = ',sumsq1)
    sumsq2 = 0
    for i in range(6):
        sumcol = 0
        for j in range(4):
            sumcol += df[i][j]
        sumsq2 += sumcol**2
    sumsq2 = sumsq2/4
    print('Q2 = ',sumsq2)
    sumsq3 = 0
    for i in range(4):
        sumcol = 0
        for j in range(6):
            sumcol += df[j][i]
        sumsq3 += sumcol**2
    sumsq3 = sumsq3/6
    print('Q3 = ', sumsq3)
    sumsq4 = 0
    for i in range(6):
        sumcol = 0
        for j in range(4):
            sumcol += df[i][j]
        sumsq4 += sumcol
    sumsq4 = sumsq4**2/24
    print('Q4 = ',sumsq4)
    s0_sq = (sumsq1+sumsq4-sumsq2-sumsq3)/15
    sa_sq = (sumsq2-sumsq4)/5
    sb_sq = (sumsq3-sumsq4)/3
    fisher1 = 2.9
    fisher2 = 3.29
    if sa_sq/s0_sq < fisher1:
        print('Фактор А значущій')
    else:
        print('Фактор А не значуій')
    if sb_sq/s0_sq < fisher2:
        print('Фактор Б значущій')
    else:
        print('Фактор Б не значуій')
    
    
    
            
    
    


a = pd.read_csv('kpi5.txt', sep='\s+', header=None) 
dataset = a.loc[:, a.columns != 0]        #считывание датасета
dataset.columns = [ i for i in range(6)]   #нумерация столбцов

result_stats = prep(dataset)
print('Массив середніх значень за згруповуванням: ')
print(result_stats)
result_dataset = []

twoway_anova(result_stats)






