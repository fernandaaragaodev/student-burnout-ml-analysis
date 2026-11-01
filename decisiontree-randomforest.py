import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('ai_student_impact_dataset (1).csv')


if len(df) > 5000:
  df = df.sample(n=5000, random_state=42).reset_index(drop=True)


if 'Student_ID' in df.columns:
  df = df.drop(columns=['Student_ID'])

target = 'Burnout_Risk_Level'

# Pre-processamento dos dados escolHidos
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


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dicionario para guardar a acuracia media de cada modelo
resultados = {}

print('=== EXECUTANDO AVALIACAO DOS ALGORITMOS (5-FOLD CV) ===\n')

# 3. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
scores_dt = cross_val_score(dt, X_scaled, y, cv=5, scoring='accuracy')
resultados['Decision Tree'] = scores_dt.mean()
print(
    f'Decision Tree -> Média: {scores_dt.mean():.4f} | Desvio:'
    f' {scores_dt.std():.4f}'
)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
scores_rf = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')
resultados['Random Forest'] = scores_rf.mean()
print(
    f'Random Forest -> Média: {scores_rf.mean():.4f} | Desvio:'
    f' {scores_rf.std():.4f}'
)

# 5. KNN (Testando K=15 que teve o melhor resultado na atv passada)
knn = KNeighborsClassifier(n_neighbors=15)
scores_knn = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
resultados['KNN (K=15)'] = scores_knn.mean()
print(
    f'KNN (K=15)     -> Média: {scores_knn.mean():.4f} | Desvio:'
    f' {scores_knn.std():.4f}'
)

#Naive Bayes
nb = GaussianNB()
scores_nb = cross_val_score(nb, X_scaled, y, cv=5, scoring='accuracy')
resultados['Naive Bayes']  = scores_nb.mean()
print(
    f'Naive Bayes   -> Média: {scores_nb.mean():.4f} | Desvio:'
    f' {scores_nb.std():.4f}'
)


melhor_modelo = max(resultados, key=resultados.get)
melhor_acuracia = resultados[melhor_modelo]

print('\n' + '=' * 40)
print(
    f'MELHOR ALGORITMO: {melhor_modelo} com acurácia de'
    f' {melhor_acuracia * 100:.2f}%'
)
print('=' * 40)
