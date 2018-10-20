# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:23:42 2018

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

ds = []

with driver.session() as session:
    session.read_transaction(load_dataset, ds)

labels = ["player_id", "name", "wins", "loss", "total_matches", "min_year", "max_year", "years_exp", "percent_wins"]
dataset = pd.DataFrame(data = ds, columns=labels)

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


X = dataset.iloc[:, 7:8].values
y = dataset.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Wins% vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Wins (%)')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.scatter(X_test, y_pred, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Wins% vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Wins (%)')
plt.show()

regressor.predict([[1.0]])