import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('P1_diamonds.csv')

# ~ print(df.head(10).to_string())

# Удаление Unnamed сещлбца, axis = 1 означает, что мы удаляем столбец
df = df.drop(['Unnamed: 0'], axis = 1)

# Создание переменных для категорий
categorical_features = ['cut', 'color', 'clarity']
le = LabelEncoder()

# Замена категорий на численные значения
for i in range(3):
	new = le.fit_transform(df[categorical_features[i]])
	df[categorical_features[i]] = new
	
print(df.head(10).to_string())
