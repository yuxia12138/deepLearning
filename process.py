import pandas as pd
import numpy as np

data_path = r'D:\workspace\temp\Book1.xlsx'
data = pd.read_excel(data_path,header=0)
# data = data.drop([0])
data_save = []
slice = []

for index, row in data.iterrows():       
    print(row)
    if pd.isnull(row).any() :
        data_save.append(slice)
        slice = []

    else:
        slice.append(row)
print()