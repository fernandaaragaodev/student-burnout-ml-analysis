import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

df = pd.read_csv('ai_student_impact_dataset (1).csv')

le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col].astype(str))

df = df.fillna(df.median())

coluna_alvo = df.columns[-1]  
X = df.drop(columns=[coluna_alvo])
y = df[coluna_alvo]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_treino = scaler.fit_transform(X_treino)
X_teste = scaler.transform(X_teste)

rede = MLPClassifier(max_iter=3000, random_state=42)

parametros = {
    'hidden_layer_sizes': [
        (),
        (10,),
        (20,),
        (10, 10),
        (20, 10),
        (50, 25)
    ],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'learning_rate_init': [0.001, 0.01]
}

busca = GridSearchCV(
    estimator=rede,
    param_grid=parametros,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

busca.fit(X_treino, y_treino)

print("Melhores parâmetros:", busca.best_params_)
print("Melhor acurácia na validação:", busca.best_score_)
print("Acurácia no teste:", busca.score(X_teste, y_teste))

y_pred = busca.predict(X_teste)
print(classification_report(y_teste, y_pred))
