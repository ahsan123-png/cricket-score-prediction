import streamlit as st
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the pickle file
pickle_file_path = os.path.join(current_dir, 'pipe.pkl')

# Load the pre-trained pipeline
with open(pickle_file_path, 'rb') as file:
    pipe = pickle.load(file)

# Define the list of teams and cities
teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 'England', 'West Indies',
         'Afghanistan', 'Pakistan', 'Sri Lanka']
cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 'London', 'Pallekele',
          'Barbados', 'Sydney', 'Melbourne', 'Durban', 'St Lucia', 'Wellington', 'Lauderhill',
          'Hamilton', 'Centurion', 'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton',
          'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 'Nagpur', 'Chandigarh',
          'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff', 'Christchurch', 'Trinidad']

# Set the title of the app
st.title('Cricket Score Predictor')

# Create columns for the input fields
col1, col2 = st.columns(2)

# Input fields for batting and bowling teams
with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

# Input field for city
city = st.selectbox('Select city', sorted(cities))

# Create more columns for additional input fields
col3, col4, col5 = st.columns(3)

# Input fields for current score, overs, and wickets
with col3:
    current_score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs done (works for over > 5)')
with col5:
    wickets = st.number_input('Wickets out')

# Input field for runs scored in the last 5 overs
last_five = st.number_input('Runs scored in last 5 overs')

# Prediction button
if st.button('Predict Score'):
    # Calculate additional features
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs

    # Create a DataFrame for the input features
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'current_score': [current_score],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'crr': [crr],
        'last_five': [last_five]
    })

    # Predict the score using the pre-trained pipeline
    result = pipe.predict(input_df)

    # Display the predicted score
    st.header("Predicted Score - " + str(int(result[0])))
