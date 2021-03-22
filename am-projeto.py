import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
np.random.seed(0)# seed para determinismo
folder = 'ml-latest-small'

#leitura de arquivos
ratings = pd.read_csv(f"{folder}/ratings.csv")

#cria uma coluna onde com valor True se rating >= 3, Falso caso contrário e deleta o rating numérico
ratings['liked'] = ratings['rating'] >= 3.
ratings = ratings.drop(['rating'], axis = 1)

#para cada usuário, o último filme visto será o conjunto de testes, e os outros o conjunto de treinamento

ratings['last_movie'] = False #cria uma coluna para gravar qual o último filme para cada usuário
users = pd.unique(ratings['userId'])
for user in users:
    user_entries = ratings.loc[ratings['userId'] == user]
    last_timestamp = max(user_entries['timestamp'])
    ratings.loc[(
        (ratings['userId'] == user) & 
        (ratings['timestamp'] == last_timestamp)), 
        'last_movie'] = True


#cria conjuntos de treinamento e teste
train = ratings.loc[ratings['last_movie'] == False]
test = ratings.loc[ratings['last_movie'] == True]
train = train.drop(['last_movie'], axis=1)
test = test.drop(['last_movie'], axis=1)

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
