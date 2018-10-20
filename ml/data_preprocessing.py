# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 12:10:44 2018

@author: Sergio
"""
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from datetime import datetime

#Para instalar o driver do neo4j
#!pip install neo4j-driver
from neo4j.v1 import GraphDatabase

user = "neo4j"
passwd = "4joen"
driver = GraphDatabase.driver("bolt://localhost:7687", auth=(user,passwd))

def load_dataset(tx, ds):
    for record in tx.run("match (p:Player)-[:WIN]-(m:Match) "
                         "with p, count(m) as wins, "
                         "min(date(m.tourney_date).year) as min_year_w, "
                         "max(date(m.tourney_date).year) as max_year_w "
                         "match (p)-[:LOSE]-(n:Match) "
                         "with p, wins, count(n) as loss, "
                         "min_year_w, max_year_w, "
                         "min(date(n.tourney_date).year) as min_year_l, "
                         "max(date(n.tourney_date).year) as max_year_l "
                         "with p, wins, loss, wins+loss as total_matches, "
                         "min_year_w, max_year_w, "
                         "min_year_l, max_year_l, "
                         "case max_year_w>max_year_l "
                         "  when true then max_year_w "
                         "  else max_year_l  "
                         "end as max_year, "
                         "case min_year_w<min_year_l "
                         "  when true then min_year_w "
                         "  else min_year_l "
                         "end as min_year "
                         "return p.player_id, p.name, wins, loss, total_matches, min_year, max_year, "
                         "max_year - min_year + 1  as years_exp,  "
                         "toFloat(wins) / toFloat(total_matches) as percent_wins "
                         "order by p.name "
                         #"limit 5"
                         ):
        #print(record)
        ds.append(record)

def load_matches(tx, ds):
    for record in tx.run("match (m:Match) return m"):
        #print(record)
        ds.append(record)

ds = []

def pontuacao(posicao):
    return 18157 * float(posicao) ** -0.779

pos = list(range(1,100))
points = [pontuacao(it) for it in pos]

plt.scatter(pos, points, color = 'red')
plt.plot(pos, points, color = 'blue')
plt.title('Points vs ranking')
plt.xlabel('ranking position')
plt.ylabel('Points')
plt.show()


with driver.session() as session:
    session.read_transaction(load_dataset, ds)

with driver.session() as session:
    session.read_transaction(load_matches, ds)

labels = list(ds[0].value(0).keys())
len(labels)

max_labels = 0
max_i = -1
for i in range(0,len(ds)):
    L = len(ds[i].value(0).keys())
    if (L > max_labels):
        max_labels = L
        max_i = i
print(max_labels, max_i)        
labels = list(ds[max_i].value(0).keys())

for i in range(0,len(ds)):
    row = []
    for l in labels:
        v = ds[i].value(0).get(l)
        row.append(v)
    ds[i] = row

matches_dataset = pd.DataFrame(data = ds, columns=labels)
matches_dataset.to_csv("matches.csv", header=labels)

labels = ["player_id", "name", "wins", "loss", "total_matches", "min_year", "max_year", "years_exp", "percent_wins"]
dataset = pd.DataFrame(data = ds, columns=labels)

dataset.to_csv("winners.csv", header=labels)
"""
NEO4J

// Importar dados de arquivo CSV
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/stakah/tennis_atp/master/atp_matches_2017.csv" AS row
CREATE (n:Match)
SET n = row

// Players e Matches de 2017
match (w:Player)-[:WIN]-(m:Match)-[:LOSE]-(v:Player)
where date(m.tourney_date).year = 2017
return w,m,v

// Cria relacionamento de Winners de Matches de 2017
match (m:Match), (p:Player)
where date(m.tourney_date).year = 2017
and m.winner_id = p.player_id
merge (m)-[:WIN]->(p)

// Cria relacionamento de Losers de Matches de 2017
match (m:Match), (p:Player)
where date(m.tourney_date).year = 2017
and m.loser_id = p.player_id
merge (m)<-[:LOSE]-(p)

// Obtem os wins e losses dos players
match (p:Player)-[:WIN]-(m:Match)
with p, count(m) as wins, 
min(date(m.tourney_date).year) as min_year_w,
max(date(m.tourney_date).year) as max_year_w
match (p)-[:LOSE]-(n:Match)
with p, wins, count(n) as loss, 
min_year_w, max_year_w, 
min(date(n.tourney_date).year) as min_year_l,
max(date(n.tourney_date).year) as max_year_l
with p, wins, loss, wins+loss as total_matches,
min_year_w, max_year_w,
min_year_l, max_year_l,
case max_year_w>max_year_l
  when true then max_year_w
  else max_year_l 
end as max_year,
case min_year_w<min_year_l
  when true then min_year_w
  else min_year_l
end as min_year
return p, wins, loss, total_matches, min_year, max_year,
max_year - min_year + 1  as years_exp, 
toFloat(wins) / toFloat(total_matches) as percent_wins
order by p.name

"""

def parsedate(d):
    try:
        dt = datetime.strptime(d, "%Y%m%d")
    except ValueError as e:
        dt = d
    return dt
        
# Importing the dataset
dataset = pd.read_csv('../atp_matches_2018.csv')
players_ds = pd.read_csv('../atp_players.csv', 
                         header=None, 
                         names=['player_id', 'first_name', 'last_name',' hand', 'birth_date', 'country_code'],
                         encoding = 'ISO-8859-1',
                         parse_dates=[4],
                         date_parser=parsedate)

for it in dataset.values:
    print(it[2])
    
cols = dataset.columns.tolist()
#match_cols = cols[:8] + cols[13:14] + cols[17:18] + cols[23:24] + cols[27:31]
#match_dataset = dataset.loc[:, match_cols]

#match_cols_min = cols[0:1] + cols[6:8] + cols[14:15] + cols[17:18] + cols[24:25] + cols[27:31]
#match_dataset_min = dataset.loc[:, match_cols_min]


def split_to_tuple(x):
    rx = re.compile(r'(\d+)(\((\d+)\))?')
    v = x.split(' ')
    v.extend(('-','-','-'))
    w = []
    for s in v:
        a = s.replace('[','').replace(']','')
        a = a.split('-')
        win = a[0]
        los = ''
        tie = ''
        if len(a) == 2:
          arr = rx.search(a[1])
          if arr:
              los = arr.group(1)
              tie = arr.group(3)
        w.extend((win, los, tie))
    return tuple(w[0:9])

scores = dataset.iloc[:, 27:28].applymap(split_to_tuple)

set_scores = pd.DataFrame(scores['score'].tolist(), columns=['set1W','set1L','set1T',
                                                             'set2W','set2L','set2T',
                                                             'set3W','set3L','set3T'])


win_cols = cols[0:1] + cols[6:8] + cols[13:15] + cols[28:31]
win_dataset = dataset.loc[:, win_cols]
win_dataset['win'] = True

#los_cols = cols[0:1] + cols[6:7] + cols[17:18] + cols[23:25] + cols[27:31]
#los_dataset = dataset.loc[:, los_cols]
#los_dataset['win'] = False

player_dataset = 

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

df = pd.DataFrame({'B': [1, 2, 3], 'C': [4, 5, 6]})
cols = df.columns.tolist()
idx = [[0]]
new_cols = [[7, 8, 9],[10,11,12]]  # can be a list, a Series, an array or a scalar   
df.insert(loc=idx, column=['A','Z'], value=new_cols)


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
