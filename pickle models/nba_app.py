import streamlit as st
import pickle
import numpy as np
from PIL import Image
# Header
# st.title('')

col1, col2, col3 = st.beta_columns(3)
# col1 = st.title('NBA Player')
image = Image.open('../Visuals/nba_logo.jpeg')
col2.image(image)
# image2 = Image.open('../Visuals/nba_logo.jpeg')
# col3.image(image2)


page = st.sidebar.selectbox('Select a category:',
('About','Offense','Defense', 'Pure Offense','Traditional')
)

if page=='About':
    st.write('This app was developed for passionate NBA Fans! To aquire the top 10 similar players based on stats simply click on the arrow to the left and provide a 2021 NBA players name in any category. Offensive category was created by identifying any statistic that highlighted a players performance while in the game. Defensive category was created by identifying any statistic that highlighted a players performance while in the game. Pure Offense describes a players scoring performance based on statistics that highlighted the ability to score only. Traditional category is described by simple box office statistics. As current the application can be used for offseason team building.')

if page=='Offense':
    player = st.text_input("Enter 2021 NBA player name:")
    if player is not "":
        st.write(f'Top 10 players offensively similar {player}:')
        off_players = pickle.load(open('off_k_stat.p', mode='rb'))
        players = off_players[player].sort_values()[1:11].index
        count=1
        for i in players:
            st.write(count, i)
            count+=1



if page=='Defense':
    player = st.text_input("Enter 2021 NBA player name:")
    if player is not "":
        st.write(f'Top 10 players defensively similar {player}')
        off_players = pickle.load(open('def_stat.p', mode='rb'))
        players = off_players[player].sort_values()[1:11].index
        count=1
        for i in players:
            st.write(count, i)
            count+=1

if page=='Pure Offense':
    player = st.text_input("Enter 2021 NBA player name:")
    if player is not "":
        st.write(f'Top 10 players who score similar to {player}')
        off_players = pickle.load(open('pure_k_stat.p', mode='rb'))
        players = off_players[player].sort_values()[1:11].index
        count=1
        for i in players:
            st.write(count, i)
            count+=1

if page=='Traditional':
    player = st.text_input("Enter 2021 NBA player name:")
    if player is not "":
        st.write(f'Top 10 players similar to {player}')
        off_players = pickle.load(open('trad_stat.p', mode='rb'))
        players = off_players[player].sort_values()[1:11].index
        count=1
        for i in players:
            st.write(count, i)
            count+=1
