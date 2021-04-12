import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
np.random.seed(0)# seed para determinismo
folder = 'ml-full'

#leitura de arquivos
ratings = pd.read_csv(f"{folder}/ratings.csv")

#cria uma coluna onde com valor True se rating >= 3, Falso caso contrário e deleta o rating numérico
ratings['liked'] = ratings['rating'] >= 3.
ratings = ratings.drop(['rating'], axis = 1)

#para cada usuário, o último filme visto será o conjunto de testes, e os outros o conjunto de treinamento

#retorna as linhas referentes ao último filme que recebeu nota de cada usuário
last_ratings = ratings.groupby('userId', as_index=False).nth(-1)

#cria uma maścara para test e train
test_mask = ratings.index.isin(last_ratings.index)
train_mask = ~test_mask

#usa maścaras
test = ratings[test_mask]
train = ratings[train_mask]

#separa entre X (atributos) e y (classes)
X_train = train.loc[:, train.columns != 'liked']
y_train = train.loc[:, ['liked']]
X_test = test.loc[:, test.columns != 'liked']
y_test = test.loc[:, ['liked']]

#treina árvore de decisão
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

#score
score = tree.score(X_test, y_test) 
print(score)
