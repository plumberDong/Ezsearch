import pandas as pd
dat_path = 'datasets/shxyj_2023_6.csv'
# reading the xls file into a pandas dataframe
df = pd.read_csv(dat_path, encoding = 'gbk')
print(df)