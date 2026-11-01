import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv(
    r'C:\Users\Usuário\Downloads\paula_fernanda_atv_03\ai_student_impact_dataset (1).csv'
)


if len(df) > 5000:
  df = df.sample(n=5000, random_state=42).reset_index(drop=True)

if 'Student_ID' in df.columns:
  df = df.drop(columns=['Student_ID'])

target = 'Burnout_Risk_Level'


X = df.drop(columns=[target]).copy()
y = df[target].copy()

col_num = X.select_dtypes(include=[np.number]).columns
col_cat = X.select_dtypes(exclude=[np.number]).columns

for col in col_num:
  X[col] = X[col].fillna(X[col].mean())

for col in col_cat:
  X[col] = X[col].fillna(X[col].mode()[0])


le = LabelEncoder()
for col in col_cat:
  X[col] = le.fit_transform(X[col].astype(str))

y = le.fit_transform(y.astype(str))


X = X.astype(float)

#Normalizacao min-max
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


print('|*|*| RESULTADOS KNN |*|*|')
ks = [1, 3, 5, 7, 9, 11, 15]

for k in ks:
  knn = KNeighborsClassifier(n_neighbors=k)
  res = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
  print(f'K = {k}')
  print('Media:', round(res.mean(), 4))
  print('Desvio:', round(res.std(), 4))
  print('-' * 20)


print('\n|*|*| RESULTADOS NAIVE BAYES |*|*|')
nb = GaussianNB()
res_nb = cross_val_score(nb, X_scaled, y, cv=5, scoring='accuracy')
print('Media NB:', round(res_nb.mean(), 4))
print('Desvio NB:', round(res_nb.std(), 4))