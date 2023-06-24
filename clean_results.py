import pandas as pd

#setear la precision para cada resultado
df = pd.read_csv('./tests/results.csv')

df.to_csv('./results_with_precision.csv', float_format='%.4f',index=False)
