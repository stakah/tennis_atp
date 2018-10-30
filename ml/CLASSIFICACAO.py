# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 08:53:28 2018

@author: Sergio
"""

# TensorFlow and tf.keras
import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow import keras

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from datetime import datetime



# Helper libraries
#import numpy as np
#import matplotlib.pyplot as plt

print(tf.__version__)


matches_ds = pd.read_csv('matches.csv')

#matches_ds['win'] = 0

#Xl = matches_ds.loc[:, ['loser_age', 'loser_ht', 'loser_rank', 'round', 'surface']].values
#Xw = matches_ds.loc[:, ['winner_age', 'winner_ht','winner_rank', 'round', 'surface']].values
#histl = []

class Round:
    """
      Classe para calcular o historico por round.
      n: Numero de jogos
      w: Numero de vitorias
      l: Numero de derrotas
    """
    def __init__(self):
        self.n = 0
        self.w = 0
        self.l = 0
    
class Surface:
    """
      Classe para calcular o historico por tipo de superficie de quadra.
      n: Numero de jogos
      w: Numero de vitorias
      l: Numero de derrotas
    """
    def __init__(self):
        self.n = 0
        self.w = 0
        self.l = 0
    
class Hist:
    """
      Classe para calcular o historico do jogador.
      player_id: identificador do jogador
      name: nome do jogador
      age: idade do jogador
      height: altura do jogador
      rank: ranking do jogador
      n: Numero de jogos
      w: Numero de vitorias
      l: Numero de derrotas
      s: dict() de Objetos com as estatisticas de jogos por tipo de superficie de quadra (Surface)
        s.n: Numero de jogos
        s.w: Numero de vitorias
        s.l: Numero de derrotas
      r: dict() de Objetos com as estatisticas de jogos por rodada (Round)
        r.n: Numero de jogos
        r.w: Numero de vitorias
        r.l: Numero de derrotas
    """
    
    def __init__(self):
        self.player_id = 0
        self.name = ''
        self.age = 0.0
        self.height = 0.0
        self.rank = 0
        self.tourney_date = 0
        self.n = 0
        self.w = 0
        self.l = 0
        self.s = dict()
        self.r = dict()
    
def get_surfaces(matches):
    l = dict()
    for [surface] in matches.loc[:, ['surface']].values:
        if isinstance(surface, float):
            surface = 'None'
        if not surface in l.keys():
            l[surface] = 1
            print(surface)
    return l

surfaces_list = get_surfaces(matches_ds)

def get_historico(matches):
    def set_player(h, p_id, p_name, p_age, p_ht, p_rank, p_tourney_date):
        if not p_id in h.keys():
            h[p_id] = Hist()
        h[p_id].player_id = p_id
        h[p_id].name = p_name
        h[p_id].age = p_age
        h[p_id].height = p_ht
        if h[p_id].tourney_date == 0:
            h[p_id].tourney_date = p_tourney_date
            h[p_id].rank = p_rank
        if h[p_id].tourney_date < p_tourney_date:
            h[p_id].tourney_date = p_tourney_date
            h[p_id].rank = p_rank
    
    def set_play_count(hh, inc_n, inc_w, inc_l):
        hh.n += inc_n
        hh.w += inc_w
        hh.l += inc_l
    
    def set_round_count(r, inc_n, inc_w, inc_l):
        r.n += inc_n
        r.w += inc_w
        r.l += inc_l
        
    def set_surface_count(s, inc_n, inc_w, inc_l):
        s.n += inc_n
        s.w += inc_w
        s.l += inc_l
            
    h = dict()
    matches['winner_hist'] = 0.0
    matches['loser_hist'] = 0.0
    matches['winner_surface'] = 0.0
    matches['loser_surface'] = 0.0
    matches['winner_round'] = 0.0
    matches['loser_round'] = 0.0

    for [loser_id, loser_name, loser_age, loser_ht, loser_rank, \
         winner_id, winner_name, winner_age, winner_ht, winner_rank, \
         match_round, match_surface, tourney_date] \
    in matches.loc[:, ['loser_id','loser_name','loser_age','loser_ht','loser_rank',\
                       'winner_id','winner_name','winner_age','winner_ht','winner_rank', \
                       'round','surface', 'tourney_date']].values:
        set_player(h, loser_id, loser_name, loser_age, loser_ht, loser_rank, tourney_date)
        set_play_count(h[loser_id], 1, 0, 1)
        
        if not match_round in h[loser_id].r.keys():
            h[loser_id].r[match_round] = Round()
        set_round_count(h[loser_id].r[match_round], 1, 0, 1)
        
        if not match_surface in h[loser_id].s.keys():
            h[loser_id].s[match_surface] = Surface()
        set_surface_count(h[loser_id].s[match_surface], 1, 0, 1)
        
        set_player(h, winner_id, winner_name, winner_age, winner_ht, winner_rank, tourney_date)
        set_play_count(h[winner_id], 1, 1, 0)
        
        if not match_round in h[winner_id].r.keys():
            h[winner_id].r[match_round] = Round()
        set_round_count(h[winner_id].r[match_round], 1, 1, 0)
        
        if not match_surface in h[winner_id].s.keys():
            h[winner_id].s[match_surface] = Surface()
        set_surface_count(h[winner_id].s[match_surface], 1, 1, 0)
    return h

H = get_historico(matches_ds)

def to_array_dict(H):
    ret = []
    for r in H.keys():
        print('r=', r)
        h = H[r]
        d = h.__dict__.copy()
        d['s'] = d['s'].copy()
        d['r'] = d['r'].copy()
        for i in h.s.keys():
            d['s'][i] = h.s[i].__dict__.copy()
        for i in h.r.keys():
            d['r'][i] = h.r[i].__dict__.copy()
        print(d)
        ret.append(d)
    return ret

X_list = to_array_dict(H)

def apply_hist(matches, h):
    def calc_winner_hist(row):
        h_winner = h[row['winner_id']]
        return h_winner.w / h_winner.n
    
    def calc_loser_hist(row):
        h_loser = h[row['loser_id']]
        return h_loser.w / h_loser.n

    def calc_winner_surface(row):
        h_winner = h[row['winner_id']]
        surface = row['surface']
        try:
            s = h_winner.s[surface]
            return s.w / s.n
        except:
            return 0.0
    
    def calc_loser_surface(row):
        h_loser = h[row['loser_id']]
        surface = row['surface']
        try:
            s = h_loser.s[surface]
            return s.w / s.n
        except:
            return 0.0
    
    def calc_winner_round(row):
        h_winner = h[row['winner_id']]
        ro = row['round']
        try:
            r = h_winner.r[ro]
            return r.w / r.n
        except:
            return 0.0
    
    def calc_loser_round(row):
        h_loser = h[row['loser_id']]
        ro = row['round']
        try:
            r = h_loser.r[ro]
            return r.w / r.n
        except:
            return 0.0
    
    matches['winner_hist'] = matches.apply(calc_winner_hist, axis=1)
    matches['winner_surface'] = matches.apply(calc_winner_surface, axis=1)
    matches['winner_round'] = matches.apply(calc_winner_round, axis=1)
    matches['loser_hist'] = matches.apply(calc_loser_hist, axis=1)
    matches['loser_surface'] = matches.apply(calc_loser_surface, axis=1)
    matches['loser_round'] = matches.apply(calc_loser_round, axis=1)

apply_hist(matches_ds, H)

import random
def get_X(matches):
    X = []
    for [w_id, w_name, w_rank, w_hist, w_sur, w_rou, \
         l_id, l_name, l_rank, l_hist, l_sur, l_rou, tourney_date] \
         in matches.loc[:, ['winner_id','winner_name','winner_rank', 'winner_hist', 'winner_surface', 'winner_round', \
                            'loser_id','loser_name','loser_rank', 'loser_hist', 'loser_surface', 'loser_round',\
                            'tourney_date']].values:
        if (random.randint(1,2) == 1):
            X.append({'id1':w_id, 'name1':w_name, 'rank1':w_rank, 'his1':w_hist, 'sur1':w_sur, 'rou1':w_rou, \
                      'id2':l_id, 'name2':l_name, 'rank2':l_rank, 'his2':l_hist, 'sur2':l_sur, 'rou2':l_rou, \
                      'p1':1, 'p2':0, 'tourney_date':tourney_date})
        else:
            X.append({'id1':l_id, 'name1': l_name, 'rank1':l_rank, 'his1':l_hist, 'sur1':l_sur, 'rou1':l_rou, \
                      'id2':w_id, 'name2': w_name, 'rank2':w_rank, 'his2':w_hist, 'sur2':w_sur, 'rou2':w_rou, \
                      'p1':0, 'p2':1, 'tourney_date':tourney_date})
    return X

#X = matches_ds.loc[:, ['winner_rank','winner_hist','winner_surface','winner_round', \
#                       'loser_rank','loser_hist','loser_surface','loser_round']]
#y = [1 for _ in X.iloc[:,0]]

Xh = get_X(matches_ds)
X = pd.DataFrame.from_dict(Xh)
y = X.loc[:, ['p1','p2']].values
X = X.loc[:, ['rank1','his1','sur1','rou1',\
              'rank2','his2','sur2','rou2']]

#yl = [0 for _ in Xl[:,0]]
#yw = [1 for _ in Xl[:,0]]

#dfl = pd.DataFrame(data = Xl, columns=['age', 'height', 'rank', 'round', 'surface'])
#dfw = pd.DataFrame(data = Xw, columns=['age', 'height', 'rank', 'round', 'surface'])
#dyl = pd.DataFrame(data = yl, columns=['win'])
#dyw = pd.DataFrame(data = yw, columns=['win'])

#X = pd.concat([dfl, dfw])
#y = pd.concat([dyl, dyw])


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis = 0)
imputer = imputer.fit(X.loc[:, ['rank1','rank2']])
X.loc[:, ['rank1','rank2']] = imputer.transform(X.loc[:, ['rank1','rank2']])


#X.iloc[:, 4] = ['None' if isinstance(s, float) else s for s in X.iloc[:,4] ]

#def find_float(X, col):
#    for r in X.iloc[:,col]:
#        if isinstance(r,float):
#            print(r)

#def find_nan(X, col):
#    for r in X.iloc[:,col]:
#        if np.isnan(r):
#            print(r)
            

# Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#labelencoder_yl = LabelEncoder()
#yl = labelencoder_yl.fit_transform(yl)

#labelencoder_X_round = LabelEncoder()
#X.iloc[:, 3] = labelencoder_X_round.fit_transform(X.iloc[:, 3])

#labelencoder_X_surface = LabelEncoder()
#X.iloc[:, 4] = labelencoder_X_surface.fit_transform(X.iloc[:, 4])

#from sklearn.feature_extraction import DictVectorizer, FeatureHasher

#fh_round = FeatureHasher(n_features=15, input_type='string')
#Xl[:, 3] = fh_round.fit_transform(Xl[:, 3])

#fh_surface = FeatureHasher(n_features=5, input_type='string')
#Xl[:, 4] = fh_surface.fit_transform(Xl[:, 4])

#onehotencoder = OneHotEncoder(categorical_features=[3,4])
#X = onehotencoder.fit_transform(X).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Tensorflow NN model
def create_model():
    model = keras.Sequential([
            keras.layers.Dense(16, input_shape=(8,), activation=tf.nn.relu),
            keras.layers.Dense(16),
            keras.layers.Dense(2, activation=tf.nn.softmax)
            ])
    
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  #loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model()

model.summary()

# Train the NN
model.fit(X_train, y_train, epochs=8)

#Install tensorflowjs
#!pip install tensorflowjs
import tensorflowjs as tfjs
# Save the model and weights
tfjs.converters.save_keras_model(model, './tfjsmodel')
model.save('./my_checkpoint')
#model.save_weights('./my_checkpoint')

# Restore the weights
#model = create_model()
#model.load_weights('./my_checkpoint')

# Evaluate accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)

# Make predictions
predictions = model.predict(X_test)

# Predictions size
len(predictions)

# First prediction
predictions[0]

# Which player would win
np.argmax(predictions[0])

# Compare with the actual player from the test set
y_test[0]

# The training data
X_test.iloc[0]

# The players and data for that prediction
Xh[132707]

#sess = tf.InteractiveSession()
#confusion = tf.confusion_matrix(pd.DataFrame(y_test).iloc[:,0].values, pd.DataFrame(predictions).iloc[:,0].values)
#confusion.eval()



def find_id_by_name(X, name):
    """
    retorna um dictionary com id e name do jogador a partir do nome do jogador.
    """
    for r in H.keys():
        h = H[r]
        if h.name.lower() == name.lower():
            return {'id': h.player_id, 'name': h.name}
       
# List dos 21 jogadores com maior quantidades de vitorias em torneios 
players = ['Jimmy Connors', 'Roger Federer', 'Ivan Lendl', 'John McEnroe', 'Rafael Nadal', 'Novak Djokovic',\
           'Pete Sampras', 'Bjorn Borg', 'Guillermo Vilas', 'Andre Agassi', 'Ilie Nastase', 'Boris Becker',\
           'Rod Laver', 'Andy Murray', 'Thomas Muster', 'Stefan Edberg', 'Stan Smith', 'Michael Chang', \
           'Arthur Ashe', 'Ken Rosewall', 'Mats Wilander']

# Popula lista auxiliar com os jogadores com maior quantidade de vitorias
top_players = []
for p in players:
    h1 = find_id_by_name(X, p)
    top_players.append(h1)

# Gera um dataframe e salva em arquivo CSV
df = pd.DataFrame(top_players)
df.to_csv("top_players.csv")

# Matriz de entrada para fazer os matches hipoteticos com os top 21
X_top = []
for i in range(0, len(top_players)-1):
    for j in range (i+1, len(top_players)):
        id1 = top_players[i]['id']
        id2 = top_players[j]['id']
        p1 = H[id1]
        p2 = H[id2]
        X_top.append({'id1':id1, 'name1':p1.name, 'rank1':p1.rank, 'his1':p1.w/p1.n, 'sur1':0.5, 'rou1':0.5, \
                      'id2':id2, 'name2':p2.name, 'rank2':p2.rank, 'his2':p2.w/p2.n, 'sur2':0.5, 'rou2':0.5, })

top_ds = pd.DataFrame(X_top)

imputer = imputer.fit(top_ds.loc[:, ['rank1','rank2']])
top_ds.loc[:, ['rank1','rank2']] = imputer.transform(top_ds.loc[:, ['rank1','rank2']])

X_top = top_ds.loc[:, ['rank1','his1','sur1','rou1','rank2','his2','sur2','rou2']]

top_predictions = model.predict(X_top)

top_result = pd.DataFrame(top_ds.loc[:,['id1','name1','id2','name2']])

top_result['p1'] = 0
top_result['p2'] = 0

top_result.loc[:,['p1','p2']] = top_predictions

top_result.to_csv('top_result.csv')
