import streamlit as st
import pandas as pd
from data_handler import DataHandler
from models import PredictionModel, OddsBasedModel, StatisticalModel
from utils import (create_probability_chart, create_form_chart,
                  create_history_table, calculate_combined_prediction)

# Initialize session state
if 'data_handler' not in st.session_state:
    st.session_state.data_handler = DataHandler("c417cb5967msh54c12f850f3798cp12a733jsn315fab53ee3c")
    
if 'models' not in st.session_state:
    st.session_state.prediction_model = PredictionModel()
    st.session_state.odds_model = OddsBasedModel()
    st.session_state.statistical_model = StatisticalModel()

# Page config
st.set_page_config(page_title="Football Match Predictor", layout="wide")

# Title and description
st.title("Football Match Prediction System")
st.markdown("""
This system uses multiple analysis methods to predict football match outcomes:
- Machine Learning based prediction
- Statistical Analysis
- Odds-based calculation
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('oranlar1234.csv')
    return df

try:
    df = load_data()
    teams = sorted(df['HomeTeam'].unique())
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Team selection
col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("Select Home Team", teams)
    
with col2:
    away_teams = [team for team in teams if team != home_team]
    away_team = st.selectbox("Select Away Team", away_teams)

# Odds input
st.subheader("Betting Odds (Optional)")
odds_col1, odds_col2, odds_col3 = st.columns(3)

with odds_col1:
    home_odds = st.number_input("Home Win Odds", min_value=1.0, step=0.01)
    
with odds_col2:
    draw_odds = st.number_input("Draw Odds", min_value=1.0, step=0.01)
    
with odds_col3:
    away_odds = st.number_input("Away Win Odds", min_value=1.0, step=0.01)

# Make predictions when teams are selected
if home_team and away_team:
    try:
        # Train models if not already trained
        if not hasattr(st.session_state.prediction_model, 'is_trained'):
            with st.spinner("Training prediction models..."):
                st.session_state.prediction_model.train(df)
                st.session_state.prediction_model.is_trained = True
        
        # Get predictions from each model
        features = st.session_state.prediction_model.prepare_features(df, home_team, away_team)
        ml_pred = st.session_state.prediction_model.predict(features)
        
        stat_pred = st.session_state.statistical_model.calculate_probabilities(df, home_team, away_team)
        
        odds_pred = None
        if all([home_odds > 1.0, draw_odds > 1.0, away_odds > 1.0]):
            odds_pred = st.session_state.odds_model.calculate_probabilities(home_odds, draw_odds, away_odds)
        
        # Calculate combined prediction
        final_pred = calculate_combined_prediction(ml_pred, odds_pred, stat_pred)
        
        # Display predictions
        st.subheader("Match Prediction")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.plotly_chart(create_probability_chart(
                home_team, away_team, ml_pred, "Machine Learning"
            ), use_container_width=True)
            
        with col2:
            st.plotly_chart(create_probability_chart(
                home_team, away_team, stat_pred, "Statistical"
            ), use_container_width=True)
            
        with col3:
            if odds_pred is not None:
                st.plotly_chart(create_probability_chart(
                    home_team, away_team, odds_pred, "Odds-Based"
                ), use_container_width=True)
        
        # Display final combined prediction
        st.subheader("Combined Prediction")
        st.plotly_chart(create_probability_chart(
            home_team, away_team, final_pred, "Combined Model"
        ), use_container_width=True)
        
        # Display team form comparison
        st.subheader("Team Form Comparison")
        home_stats = st.session_state.prediction_model._calculate_form(
            df[df['HomeTeam'] == home_team].tail(5), home_team
        )
        away_stats = st.session_state.prediction_model._calculate_form(
            df[df['AwayTeam'] == away_team].tail(5), away_team
        )
        
        st.plotly_chart(create_form_chart(
            [home_stats, home_stats, home_stats, home_stats, home_stats],
            [away_stats, away_stats, away_stats, away_stats, away_stats],
            home_team, away_team
        ), use_container_width=True)
        
        # Display historical matches
        st.subheader("Recent Head-to-Head Matches")
        st.markdown(create_history_table(df, home_team, away_team), unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error making predictions: {e}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Data from API-Football")
