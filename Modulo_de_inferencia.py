import pandas as pd
from pickle import load

nova_paciente = pd.DataFrame([{
    'age': '50-59',
    'menopause': 'ge40',
    'tumor-size': '15-19',
    'inv-nodes': '0-2',
    'node-caps': 'no',
    'deg-malig': 1,
    'breast': 'right',
    'breast-quad': 'central',
    'irradiat': 'no'
}])

colunas_modelo = load(open('colunas_breast_cancer.model', 'rb'))
colunas_categoricas = load(open('categorias_breast_cancer.model', 'rb'))

nova_paciente[colunas_categoricas] = (
    nova_paciente[colunas_categoricas]
    .replace('?', 'desconhecido')
    .fillna('desconhecido')
)
nova_paciente = pd.get_dummies(
    nova_paciente, columns=colunas_categoricas, dtype=int
)
nova_paciente = nova_paciente.reindex(columns=colunas_modelo, fill_value=0)

normalizador = load(open('normalizador_breast_cancer.model', 'rb'))
nova_paciente = normalizador.transform(nova_paciente)
nova_paciente = pd.DataFrame(nova_paciente, columns=colunas_modelo)

breast_cancer_tree = load(open('breast_cancer.model', 'rb'))
classe_nova_paciente = breast_cancer_tree.predict(nova_paciente)
print(classe_nova_paciente[0])
