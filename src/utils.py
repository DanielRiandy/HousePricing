import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/ebooks/html/mva/mvahtmlnode11.html

def transform_(data: pd.DataFrame, column: str) -> pd.DataFrame:
    if column in ['crim', 'indus', 'nox', 'rm', 'dis', 'rad', 'tax']:
        data[column] = np.log(data[column])
    elif column == 'zn':
        data[column] = data[column]/10
    elif column == 'age':
        data[column] = (data[column]**2/5)/10000
    elif column == 'ptratio':
        data[column] = (np.exp(data[column]) * 0.4)/1000
    elif column == 'b':
        data[column] = data[column]/100
    elif column == 'lstat':
        data[column] = np.sqrt(data[column])

    return data[column]

def plot_(data: pd.DataFrame, column: str):
    ## plot 1
    plt.subplot(2,2,1)
    plt.boxplot(data[column])

    ## plot 2
    plt.subplot(2,2,2)
    plt.hist(data[column])

    ## plot 3
    temp = transform_(data=data, column=column)

    plt.subplot(2,2,3)
    plt.boxplot(temp)

    ## plot 4
    plt.subplot(2,2,4)
    plt.hist(temp)

    plt.show()

