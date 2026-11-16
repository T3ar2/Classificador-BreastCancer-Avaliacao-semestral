import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from pickle import dump

dados = pd.read_csv('breast-cancer.csv', sep=',', na_values=['?'])

dados_atributos = dados.drop(columns=['Class'])
dados_classes = dados['Class']

X_treino, X_teste, y_treino, y_teste = train_test_split(
    dados_atributos,
    dados_classes,
    test_size=0.2,
    random_state=42,
    stratify=dados_classes
)

colunas_categoricas = X_treino.select_dtypes(
    include=['object', 'string']
).columns.tolist()

X_treino[colunas_categoricas] = X_treino[colunas_categoricas].fillna('desconhecido')
X_teste[colunas_categoricas] = X_teste[colunas_categoricas].fillna('desconhecido')

X_treino = pd.get_dummies(X_treino, columns=colunas_categoricas, dtype=int)
X_teste = pd.get_dummies(X_teste, columns=colunas_categoricas, dtype=int)
X_teste = X_teste.reindex(columns=X_treino.columns, fill_value=0)

colunas_modelo = X_treino.columns.tolist()

normalizador = MinMaxScaler()
modelo_normalizador = normalizador.fit(X_treino)

dados_atributos_normalizados = modelo_normalizador.transform(X_treino)
dados_teste_normalizados = modelo_normalizador.transform(X_teste)

dados_atributos_normalizados = pd.DataFrame(
    dados_atributos_normalizados, columns=colunas_modelo, index=X_treino.index
)
dados_teste_normalizados = pd.DataFrame(
    dados_teste_normalizados, columns=colunas_modelo, index=X_teste.index
)

criterion = ['gini', 'entropy', 'log_loss']
splitter = ['best', 'random']
min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=5)]
max_depth = [2, 3, 4, 5, None]
min_samples_leaf = [1, 5, 10, 20]
# Comparar pesos normais e balanceados sem gerar pacientes sinteticas.
class_weight = [None, 'balanced']

decision_tree_grid = {
    'criterion': criterion,
    'splitter': splitter,
    'min_samples_split': min_samples_split,
    'max_depth': max_depth,
    'min_samples_leaf': min_samples_leaf,
    'class_weight': class_weight
}

tree = DecisionTreeClassifier(random_state=42)
tre_hiperparametros = RandomizedSearchCV(
    estimator=tree,
    param_distributions=decision_tree_grid,
    n_iter=60,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

tre_hiperparametros.fit(dados_atributos_normalizados, y_treino)
print('Melhores hiperparametros:', tre_hiperparametros.best_params_)
print('Acuracia de validacao:', tre_hiperparametros.best_score_)

tree = DecisionTreeClassifier(
    **tre_hiperparametros.best_params_, random_state=42
)
breast_cancer_tree = tree.fit(dados_atributos_normalizados, y_treino)

classes_preditas = breast_cancer_tree.predict(dados_teste_normalizados)
acuracia_breast_cancer_tree = accuracy_score(y_teste, classes_preditas)
print('Acuracia no teste:', acuracia_breast_cancer_tree)

dump(modelo_normalizador, open('normalizador_breast_cancer.model', 'wb'))
dump(breast_cancer_tree, open('breast_cancer.model', 'wb'))
dump(colunas_modelo, open('colunas_breast_cancer.model', 'wb'))
dump(colunas_categoricas, open('categorias_breast_cancer.model', 'wb'))
