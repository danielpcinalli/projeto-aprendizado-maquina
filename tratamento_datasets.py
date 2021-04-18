import pandas as pd
import numpy as np

folder_original = 'ml-latest-small'
folder_tratado = 'ml-small-tratado'

def ratings_ordenado():
    """
    Retorna o arquivo ratings.csv ordenado caso exista, caso contrário o cria
    """
    file_location = f'{folder_tratado}/ratings_ordenado.csv'
    try:
        return pd.read_csv(file_location)
    except:
        ratings = pd.read_csv(f"{folder_original}/ratings.csv")

        #ordena para cada usuário, o timestamp em ordem crescente
        ratings = ratings.sort_values(by=['userId', 'timestamp'])
        #retorna as linhas referentes ao último filme que recebeu nota de cada usuário
        last_ratings = ratings.groupby('userId', as_index=False).nth(-1)
        #cria uma coluna onde o último filme para cada usuário possui valor True
        isLastTimestamp = ratings.index.isin(last_ratings.index)
        ratings['isLastMovie'] = isLastTimestamp

        #retira timestamp, que não será mais utilizada
        ratings = ratings.drop(['timestamp'], axis=1)

        #salva arquivo
        ratings = ratings.to_csv(file_location, index=False)
    
    return pd.read_csv(file_location)

def tratamento_filme_classificacao_e_regressao():
    #leitura de arquivo
    ratings = ratings_ordenado()

    #cria a coluna 'liked' = 'rating' > 3.0
    ratings['liked'] = ratings['rating'] >= 3.0

    #cria conjuntos de teste e treinamento
    train = ratings[ratings['isLastMovie'] == False]
    test = ratings[ratings['isLastMovie'] == True]

    #retira coluna não usada e salva
    train.drop(['rating', 'isLastMovie'], axis=1).to_csv(f'{folder_tratado}/classificacao_filme_train.csv', index=False)
    test.drop(['rating', 'isLastMovie'], axis=1).to_csv(f'{folder_tratado}/classificacao_filme_test.csv', index=False)

    train.drop(['liked', 'isLastMovie'], axis=1).to_csv(f'{folder_tratado}/regressao_filme_train.csv', index=False)
    test.drop(['liked', 'isLastMovie'], axis=1).to_csv(f'{folder_tratado}/regressao_filme_test.csv', index=False)

def tratamento_genero():
    #leitura de arquivo
    ratings = ratings_ordenado()
    movies = pd.read_csv(f'{folder_original}/movies.csv')

    #separa cada genero em uma coluna
    genres = set()

    for list_of_genres in movies['genres']:
        list_of_genres = list_of_genres.split('|')
        genres.update(list_of_genres)
        
    for genre in genres:
        movies[genre] = movies['genres'].str.contains(genre)

    #cria a coluna 'liked' = 'rating' > 3.0
    ratings['liked'] = ratings['rating'] >= 3.0

    #retira colunas que não serão usadas
    ratings = ratings.drop(['rating'], axis=1)
    movies = movies.drop(['title', 'genres'], axis=1)

    #join entre os dataframes
    ratings_genres = ratings.merge(movies, on='movieId')

    #cria conjuntos de teste e treinamento
    train = ratings_genres[ratings_genres['isLastMovie'] == False]
    test = ratings_genres[ratings_genres['isLastMovie'] == True]

    #retira coluna não usada e salva
    train.drop(['isLastMovie'], axis=1).to_csv(f'{folder_tratado}/classificacao_genero_train.csv', index=False)
    test.drop(['isLastMovie'], axis=1).to_csv(f'{folder_tratado}/classificacao_genero_test.csv', index=False)

def tratamento_genoma():
    #leitura de arquivo
    ratings = ratings_ordenado()
    genome = pd.read_csv(f'{folder_original}/genome-scores.csv')

    #cria a coluna 'liked' = 'rating' > 3.0
    ratings['liked'] = ratings['rating'] >= 3.0

    #transforma o dataframe, colocando os genomas em colunas
    genome = genome.pivot(index='movieId', columns='tagId', values='relevance')

    #limitando aos 2 primeiros genomas, arquivo muito grande
    # genome = genome.iloc[:, 0:2] 

    #join entre os dataframes
    ratings_genome = ratings.merge(genome, on='movieId')

    #cria conjuntos de teste e treinamento
    train = ratings_genome[ratings_genome['isLastMovie'] == False]
    test = ratings_genome[ratings_genome['isLastMovie'] == True]    

    #retira coluna não usada e salva
    train.drop(['isLastMovie'], axis=1).to_csv(f'{folder_tratado}/classificacao_genoma_train.csv', index=False)
    test.drop(['isLastMovie'], axis=1).to_csv(f'{folder_tratado}/classificacao_genoma_test.csv', index=False)

if __name__ == '__main__':
    # tratamento_filme_classificacao_e_regressao()
    # tratamento_genero()
    tratamento_genoma()
