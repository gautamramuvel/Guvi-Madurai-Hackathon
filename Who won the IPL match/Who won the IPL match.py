
# coding: utf-8

# # WHO WON THE IPL MATCH?

# # Introduction
# 
# The IPL is the most highly viewed game in India. Here, we have made an attempt to predict the winner of the game given the data each ball wise, over wise and game wise.
# 
# In this notebook, we have explored whether these games are really completely random or are there any underlying latent patterns in them. We will start by fetching and preprocessing our data.

# In[1]:


import os
cwd = os.getcwd()

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np


# In[2]:


TestDeliveries = pd.read_csv(cwd+'//data//TestDeliveries.csv')
TestMatches = pd.read_csv(cwd+'//data//Testmatches.csv')
TrainDeliveries = pd.read_csv(cwd+'//data//TrainDeliveries.csv')
TrainMatches = pd.read_csv(cwd+'//data//Trainmatches.csv')


# In[3]:


print(TrainDeliveries.shape)
print(TrainMatches.shape)
print(TestDeliveries.shape)
print(TestMatches.shape)


# # PREPROCESSING DATA

# # Checking for null values

# In[4]:


print(TrainDeliveries.isnull().sum())


# In[5]:


print(TrainMatches.isnull().sum())


# In[6]:


print(TestDeliveries.isnull().sum())


# In[7]:


print(TestMatches.isnull().sum())


# # Data Cleaning

# In[8]:


TrainDeliveries.head(5)


# In[9]:


#Clean TrainDeliveries dataframe for NaN, strings

#removed Team from Team1
TrainDeliveries['batting_team'] = TrainDeliveries['batting_team'].str[4:]

#removed Team from Team1
TrainDeliveries['bowling_team'] = TrainDeliveries['bowling_team'].str[4:] 

#removed Player from Player 105
TrainDeliveries['batsman'] = TrainDeliveries['batsman'].str[7:] 

#removed Player from Player 105
TrainDeliveries['non_striker'] = TrainDeliveries['non_striker'].str[7:] 

#removed Player from Player 105
TrainDeliveries['bowler'] = TrainDeliveries['bowler'].str[7:] 

#replaced NaN values with 0
TrainDeliveries.fillna(0, inplace=True) 

TrainDeliveries.loc[TrainDeliveries.player_dismissed != 0, 'player_dismissed'] = TrainDeliveries[TrainDeliveries.player_dismissed != 0].player_dismissed.str[7:] 
TrainDeliveries.loc[TrainDeliveries.fielder != 0, 'fielder'] = TrainDeliveries[TrainDeliveries.fielder != 0].fielder.str[7:]

mapping = {'caught':1, 'bowled':2, 'run out':3,
          'lbw':4, 'caught and bowled':5, 'stumped':6,
          'retired hurt':7, 'hit wicket':8, 'obstructing the field':9}
TrainDeliveries.replace({'dismissal_kind':mapping}, inplace=True)


# In[10]:


TrainDeliveries.head()


# In[11]:


TestDeliveries.head()


# In[12]:


#clean TestDeliveries for NaN, strings

#removed Team from Team1
TestDeliveries['batting_team'] = TestDeliveries['batting_team'].str[4:]

#removed Team from Team1
TestDeliveries['bowling_team'] = TestDeliveries['bowling_team'].str[4:]

#removed Player from Player 105
TestDeliveries['batsman'] = TestDeliveries['batsman'].str[7:]

#removed Player from Player 105
TestDeliveries['non_striker'] = TestDeliveries['non_striker'].str[7:]

#removed Player from Player 105
TestDeliveries['bowler'] = TestDeliveries['bowler'].str[7:]

#replaced NaN values with 0
TestDeliveries.fillna(0, inplace=True) 
TestDeliveries.loc[TestDeliveries.player_dismissed != 0, 'player_dismissed'] = TestDeliveries[TestDeliveries.player_dismissed != 0].player_dismissed.str[7:] 
TestDeliveries.loc[TestDeliveries.fielder != 0, 'fielder'] = TestDeliveries[TestDeliveries.fielder != 0].fielder.str[7:]

mapping = {'caught':1, 'bowled':2, 'run out':3,
          'lbw':4, 'caught and bowled':5, 'stumped':6,'retired hurt':7, 'hit wicket':8, 'obstructing the field':9}
TestDeliveries.replace({'dismissal_kind':mapping}, inplace=True)

TestDeliveries['bowling_team'] = TestDeliveries['bowling_team'].replace({'4s':'4'})
TestDeliveries['batting_team'] = TestDeliveries['batting_team'].replace({'4s':'4'})


# In[13]:


TestDeliveries.head()


# In[14]:


TrainMatches.head()


# In[15]:


#clean Trainmatches dataframe for NaN, strings

TrainMatches['city'] = TrainMatches['city'].str[4:]
TrainMatches['team1'] = TrainMatches['team1'].str[4:]
TrainMatches['team2'] = TrainMatches['team2'].str[4:]
TrainMatches['toss_winner'] = TrainMatches['toss_winner'].str[4:]
TrainMatches['winner'] = TrainMatches['winner'].str[4:]
TrainMatches['player_of_match'] = TrainMatches['player_of_match'].str[7:]
TrainMatches['venue'] = TrainMatches['venue'].str[7:]

map1 = {'field':1, 'bat':2}
TrainMatches.replace({'toss_decision':map1}, inplace=True)

map2 = {'normal':1, 'tie':2, 'no result':3}
TrainMatches.replace({'result':map2}, inplace=True)

TrainMatches.rename(columns={'id':'match_id'}, inplace=True)
TrainMatches.fillna(0, inplace=True)


# In[16]:


TrainMatches.head()


# In[17]:


# get a list of the columns
col_list = list(TrainMatches)
# swap the elements
col_list[9], col_list[13] = col_list[13], col_list[9]
TrainMatches = TrainMatches.loc[TrainMatches.index[:], col_list]


# In[18]:


TrainMatches.head()


# In[19]:


TestMatches.head()


# In[20]:


#clean TestMatches for NaN, strings

TestMatches['city'] = TestMatches['city'].str[4:]
TestMatches['team1'] = TestMatches['team1'].str[4:]
TestMatches['team2'] = TestMatches['team2'].str[4:]
TestMatches['toss_winner'] = TestMatches['toss_winner'].str[4:]
TestMatches['venue'] = TestMatches['venue'].str[7:]

map1 = {'field':1, 'bat':2}
TestMatches.replace({'toss_decision':map1}, inplace=True)

map2 = {'normal':1, 'tie':2, 'no result':3}
TestMatches.replace({'result':map2}, inplace=True)


# In[21]:


TestMatches.head()


# # Features

# We will first merge the two datasets(Deliveries and Matches) for both Train and Test.
# Since we are predicting the winning team for a game, we have identified four features that don't have any foreseeable bearing on the outcome of the match. These features are:
# 
# 1. Dismissal Kind: It explains how a particular player was dismissed viz. caught, bowled, run out, lbw, caught and bowled, stumped, retired hurt, hit wicket and obstructing the field. These natures of dismissal don't affect the chances of which team wins the match. Rather it is the actual player dismissed which affects more. So we can safely drop this feature.
# 
# 2. Win by Runs: It details by how many runs was a match won. This data gives information after a team has won and so it can be safely disregarded when determining which team will win the match.
# 
# 3. Win by Wickets: It details by how many wickets was a match won. Again, this data provides information after a team has won and so it can be safely disregarded when determining which team will the match.
# 
# 4. Player of Match: It explains who was the player of the match. This information has no bearing on who will win the match so it can be again disregarded.
# 
# This leaves us with the following features in our dataset:
# 
#     1. Match ID: It is the ID of each match played.
#     
#     2. Inning: It is a division of the cricket match during which one team takes its turn to bat.
# 
#     3. Batting Team: It is the ID of the batting team in the match.
#     
#     4. Bowling Team: It is the ID of the bowling team in the match.
#     
#     5. Over: The over number of the innings.
#     
#     6. Ball: The ball number for a given over of the innings.
#     
#     7. Batsman: The Player Id for a given batsman for a given ball.
#     
#     8. Non striker: The Player Id for non-striker batsman for this ball.
#     
#     9. Bowler: The player Id of bowler for this ball.
#     
#     10. Is super over: A binary flag suggesting if this is a super over
#     
#     11. Wide Runs: Number of wide runs on this ball
#     
#     12. Bye Runs: Number of bye runs on this ball
#     
#     13. Legbye Runs: Number of legbye runs on this ball
#     
#     14. Noball Runs: Number of noball runs on this ball
#     
#     15. Penalty Runs: Number of penalty runs on this ball
#     
#     14. Batsman Runs: Number of batsman made runs on this ball
#     
#     15. Extra Runs: Total number of extra runs on this ball (sum of wide, bye, legbye, noball, and penalty)
#     
#     16. Total Runs: Total number of runs on this ball (sum of batsman_runs and extra_runs)
#     
#     17. Player Dismissed: Player dismissed on this ball if any (PlayerID)
#     
#     18. Fielder: Fielder if dismissal_kind requires (PlayerID)
#     
#     19. Season: Year of the match
#     
#     20. City: City where match was played (CityID)
#     
#     21. Team1: ID for team 1
#     
#     22. Team2: ID for team 2
#     
#     23. Toss Winner: Which team won the toss (TeamID)
#     
#     24. Toss Decision: What did the team decide to do? (field or bat)
#     
#     25. Result: What was the result (normal or tie)
#     
#     26. DL Applied: Was DL applied (binary)
#     
#     27. Venue: Which stadium the match was played (StadiumID)

# In[22]:


# Joining TrainDeliveries and TrainMatches dataframe
# Training data
data = TrainDeliveries.join(TrainMatches.set_index('match_id'), on='match_id')


# In[23]:


data.head(5)


# In[24]:


data.dtypes


# In[25]:


# Converting the datatypes into integer

data['batting_team'] = data['batting_team'].astype('int64')
data['bowling_team'] = data['bowling_team'].astype('int64')
data['batsman'] = data['batsman'].astype('int64')
data['non_striker'] = data['non_striker'].astype('int64')
data['bowler'] = data['bowler'].astype('int64')
data['player_dismissed'] = data['player_dismissed'].astype('int64')
data['fielder'] = data['fielder'].astype('int64')
data['city'] = data['city'].astype('int64')
data['team1'] = data['team1'].astype('int64')
data['team2'] = data['team2'].astype('int64')
data['toss_winner'] = data['toss_winner'].astype('int64')
data['venue'] = data['venue'].astype('int64')
data['player_of_match'] = data['player_of_match'].astype('int64')
data['winner'] = data['winner'].astype('int64')
data.drop(['dismissal_kind'], axis=1, inplace=True)


# In[26]:


data.dtypes


# In[27]:


train = data.as_matrix()
trainX = train[ : , :-4]
trainY = train[ : , -1]


# In[28]:


print(trainX.shape)
print(trainY.shape)


# In[29]:


# Joining TestDeliveries and TestMatches dataframes
# Test data
test = TestDeliveries.join(TestMatches.set_index('match_id'), on='match_id')


# In[30]:


test.head(5)


# In[31]:


test['batting_team'] = test['batting_team'].astype(str).astype(int)
test['bowling_team'] = test['bowling_team'].astype(str).astype('int64')
test['batsman'] = test['batsman'].astype('int64')
test['non_striker'] = test['non_striker'].astype('int64')
test['bowler'] = test['bowler'].astype('int64')
test['player_dismissed'] = test['player_dismissed'].astype('int64')
test['fielder'] = test['fielder'].astype('int64')
test['city'] = test['city'].astype('int64')
test['team1'] = test['team1'].astype('int64')
test['team2'] = test['team2'].astype('int64')
test['toss_winner'] = test['toss_winner'].astype('int64')
test['venue'] = test['venue'].astype('int64')
test.drop(['dismissal_kind'], axis=1, inplace=True)
test = test.as_matrix()


# In[32]:


test.shape


# # Using GridSearchCV to find best hyperparameters

# In[33]:


from sklearn.model_selection import GridSearchCV
params_grid = {
    'criterion' :['entropy','gini'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth':[5,8,12,15],
    'min_samples_leaf':[5,8,12,15]
}


# In[34]:


# training and validation set creation
from sklearn.model_selection import train_test_split
train_x, validation_x, train_y, validation_y = train_test_split(trainX, trainY, test_size=0.66, random_state = 5)


# In[35]:


# Using RandomForest model for GridCV
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1)
rfcv = GridSearchCV(estimator=rf, param_grid=params_grid, cv=5)
rfcv.fit(train_x, train_y)
print(rfcv.best_params_)
rfcv.score(validation_x, validation_y)


# In[36]:


# make model using best parameters
clf = RandomForestClassifier(criterion = 'entropy',
                             max_features = 'sqrt',
                             max_leaf_nodes = 15, 
                             n_estimators = 10, 
                             max_depth = 15,
                             min_samples_leaf = 5,
                             n_jobs =- 1)


# In[37]:


# train model
clf.fit(train_x, train_y)


# # PREDICTING THE WINNER

# In[38]:


# make predicitons on test dataset
test_data_predictions = clf.predict(test)


# In[39]:


new_test = TestDeliveries.join(TestMatches.set_index('match_id'), on='match_id')


# In[40]:


new_test.head(5)


# In[41]:


# creating dataframe for each instance of delivery
output = pd.DataFrame(new_test['match_id'])
output['winner'] = test_data_predictions


# In[42]:


# selecting last instance to find final winner
testing_df = output.copy(deep=True)
testing_df.drop_duplicates(subset='match_id', keep='last', inplace=True)
testing_df.head(5)


# In[43]:


testing_df['team1'] = TestMatches['team1'].values
testing_df['team2'] = TestMatches['team2'].values


# In[44]:


testing_df.head(10)


# In[45]:


testing_df['team1'] = testing_df['team1'].astype('int64')
testing_df['team2'] = testing_df['team2'].astype('int64')
testing_df['team_1_win_flag'] = np.where(testing_df['winner'] == testing_df['team1'],1, 0)
testing_df.head(5)


# In[46]:


subDF = pd.DataFrame({'match_id': testing_df['match_id'], 'team_1_win_flag': testing_df['team_1_win_flag']})
subDF.to_csv(cwd+'//FinalSubmission.csv', index=False)

# submitting previous run CSV file, rerun generated new file
subDF.head(10)

