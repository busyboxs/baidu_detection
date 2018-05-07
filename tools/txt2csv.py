import pandas as pd

result_txt = 'data/result.txt'
result_csv = 'data/result.csv'

all_pd = pd.read_csv(result_txt, sep=" ", header=None, names=['ImageName', 'label'])
all_pd.to_csv(result_csv, sep=" ", header=None, index=None)
