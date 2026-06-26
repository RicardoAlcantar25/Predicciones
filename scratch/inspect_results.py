import pandas as pd

df = pd.read_csv("data/results.csv")
print("Columnas de data/results.csv:")
print(df.columns.tolist())
print("\nForma del dataset:", df.shape)
print("\nPrimeras 3 filas:")
print(df.head(3))
