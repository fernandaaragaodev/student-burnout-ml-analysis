import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Carregando a base de dados
df = pd.read_csv('ai_student_impact_dataset (1).csv')

# Removendo ID do aluno se existir
if 'Student_ID' in df.columns:
  df = df.drop('Student_ID', axis=1)

# Limpando nulos caso existam
df = df.dropna().reset_index(drop=True)

# Definição do Target de 3 classes (Burnout_Risk_Level) e Features
target = 'Burnout_Risk_Level'
X_raw = df.drop(target, axis=1)
y = df[target].values

# Convertendo colunas de texto (categóricas) para numéricas com get_dummies
X_dummies = pd.get_dummies(X_raw, drop_first=True)
X_matriz = X_dummies.values

print('Formato da base após tratamento:', X_matriz.shape)
print('Distribuição das 3 classes do Target:', pd.Series(y).value_counts())


# Configurando o 5-Fold e a Grade de Parâmetros do SVM

# 5-Fold Estratificado para manter a proporção das 3 classes
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Valores de C e Kernel para explorar
param_grid = {
    'C': [0.1, 1, 10, 50, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
}


# Execução do GridSearch com Validação Cruzada (5-Fold)
print('\n-|- Rodando o GridSearch (5-Fold) para explorar o SVM -|-')

# Como o SVM é extremamente sensível à escala, foi preciso padronizar os dados por fold
# Loop do K-Fold manual para aplicar o StandardScaler de forma limpa

melhor_acuracia_geral = 0.0
melhores_params_geral = None
resultados_busca = []

# Testando cada combinação da grade de parâmetros
for kernel_teste in param_grid['kernel']:
  for c_teste in param_grid['C']:
    accs_fold = []

    for train_idx, test_idx in skf.split(X_matriz, y):
      X_train, X_test = X_matriz[train_idx], X_matriz[test_idx]
      y_train, y_test = y[train_idx], y[test_idx]

      # Padronização (StandardScaler) em cada fold de treino/teste
      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train)
      X_test_scaled = scaler.transform(X_test)

      # Criando e treinando o modelo SVC com C e Kernel da vez
      svm = SVC(C=c_teste, kernel=kernel_teste, random_state=42)
      svm.fit(X_train_scaled, y_train)

      # Predição e acurácia da dobra
      preds = svm.predict(X_test_scaled)
      accs_fold.append(accuracy_score(y_test, preds))

    acc_media = np.mean(accs_fold)
    resultados_busca.append({
        'C': c_teste,
        'kernel': kernel_teste,
        'acuracia_media': acc_media,
    })

    print(
        f'Kernel: {kernel_teste:<8} | C: {c_teste:<5} | Acurácia Média'
        f' (5-Fold): {acc_media:.4f}'
    )

    if acc_media > melhor_acuracia_geral:
      melhor_acuracia_geral = acc_media
      melhores_params_geral = {'C': c_teste, 'kernel': kernel_teste}


# Retorno dos Melhores Parâmetros e Resultado
print('\n' + '=' * 45)
print('                  RESULTADO FINAL DO SVM')
print('=' * 45)
print(f'Melhores Parâmetros Encontrados: {melhores_params_geral}')
print(f'Melhor Acurácia Média (5-Fold): {melhor_acuracia_geral:.4f}')
print('=' * 45)


# Gráfico de Comparação dos Kernels por valor de C
df_res = pd.DataFrame(resultados_busca)

plt.figure(figsize=(9, 4.5))
for k_nome in param_grid['kernel']:
  sub = df_res[df_res['kernel'] == k_nome]
  plt.plot(
      sub['C'].astype(str),
      sub['acuracia_media'],
      marker='o',
      linewidth=2,
      label=k_nome,
  )

plt.title('Impacto dos Parâmetros C e Kernel no SVM (5-Fold)')
plt.xlabel('Valor do Parâmetro C')
plt.ylabel('Acurácia Média')
plt.legend(title='Kernel')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
