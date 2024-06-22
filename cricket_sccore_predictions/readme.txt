### Project Description: Cricket Score Predictor

#### Project Overview

The Cricket Score Predictor is a machine learning application designed to forecast the final score of a cricket team in a T20 match. Utilizing a variety of input parameters such as the batting team, bowling team, match location (city), current score, number of overs completed, wickets fallen, and runs scored in the last five overs, the model provides an estimated final score. This predictive tool can be particularly useful for cricket enthusiasts, sports analysts, and bettors to gain insights during live matches.

#### Key Features

- **Team Selection**: Users can select both the batting and bowling teams from a predefined list of international cricket teams.
- **Location Selection**: The app includes a selection of cities where T20 matches are commonly played.
- **Live Match Inputs**: Inputs for current score, overs completed, wickets fallen, and runs scored in the last five overs help to provide an accurate prediction based on the current state of the match.
- **Score Prediction**: Upon receiving the inputs, the app processes the data through a pre-trained machine learning pipeline to predict the final score.

#### Technologies Used

- **Python**: The core programming language used for developing the application.
- **Streamlit**: A powerful library for creating web applications. It provides an easy way to build and deploy interactive web applications with Python.
- **Pandas and NumPy**: Essential libraries for data manipulation and numerical computations.
- **XGBoost**: An optimized gradient boosting library that provides a powerful implementation of gradient boosted decision trees designed for speed and performance.

#### Model Training

1. **Data Collection**: Historical T20 match data was collected, including features like batting team, bowling team, city, current score, overs completed, wickets fallen, and runs scored in the last five overs.
2. **Data Preprocessing**: The data was cleaned and processed to handle missing values, categorical variables were encoded, and relevant features were extracted.
3. **Feature Engineering**: Additional features such as balls left, wickets left, and current run rate (CRR) were engineered to enhance the model's predictive power.
4. **Model Selection**: XGBoost was selected for its efficiency and accuracy in handling structured data.
5. **Training and Validation**: The model was trained on a subset of the data and validated using cross-validation techniques to ensure robustness and avoid overfitting.
6. **Pipeline Creation**: A complete pipeline was created using the trained model and preprocessing steps, which was then serialized using `pickle` for deployment.

#### Deployment

The trained pipeline was integrated into a Streamlit application to make it accessible and interactive. Users can input live match data and receive predictions in real-time.

#### Streamlit App Interface

1. **Title and Introduction**: The app starts with a title "Cricket Score Predictor" and a brief description.
2. **Input Fields**:
   - **Batting Team**: Dropdown menu to select the batting team.
   - **Bowling Team**: Dropdown menu to select the bowling team.
   - **City**: Dropdown menu to select the match city.
   - **Current Score**: Numeric input for the current score of the batting team.
   - **Overs Completed**: Numeric input for the number of overs completed (must be greater than 5).
   - **Wickets Fallen**: Numeric input for the number of wickets fallen.
   - **Runs in Last 5 Overs**: Numeric input for the runs scored in the last 5 overs.
3. **Prediction Button**: A button to trigger the prediction process.
4. **Result Display**: The predicted final score is displayed prominently once the prediction is made.

#### Code Implementation

Here's the complete code for the Streamlit app:

```python
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Load the pre-trained pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

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
```

#### Running the App

To run the app locally:
1. Ensure you have Streamlit and other dependencies installed.
2. Place the `pipe.pkl` file in the same directory as your script.
3. Execute the following command in your terminal:
   ```sh
   streamlit run path_to_your_script.py
   ```
   Replace `path_to_your_script.py` with the actual path to your saved Streamlit script.

### Conclusion

This Cricket Score Predictor project leverages machine learning and modern web technologies to create an interactive and useful tool for predicting cricket scores in T20 matches. By combining historical match data with real-time inputs, users can gain insights into potential match outcomes, enhancing their viewing experience and decision-making processes.