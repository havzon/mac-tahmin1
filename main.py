import streamlit as st
import pandas as pd
import numpy as np
from data_handler import DataHandler
from models import PredictionModel, OddsBasedModel, StatisticalModel
from utils import (create_probability_chart, create_form_chart,
                  create_history_table, calculate_combined_prediction)
from strategy_advisor import StrategyAdvisor

# Initialize session state
if 'data_handler' not in st.session_state:
    st.session_state.data_handler = DataHandler("c417cb5967msh54c12f850f3798cp12a733jsn315fab53ee3c")
    st.session_state.models_trained = False

if 'models' not in st.session_state:
    st.session_state.prediction_model = PredictionModel()
    st.session_state.odds_model = OddsBasedModel()
    st.session_state.statistical_model = StatisticalModel()
    st.session_state.strategy_advisor = None

# Page config
st.set_page_config(page_title="Futbol Maç Tahmini", layout="wide")

# Title and description
st.title("Futbol Maç Tahmin Sistemi")
st.markdown("""
Bu sistem çoklu analiz yöntemleri kullanarak futbol maç sonuçlarını tahmin eder:
- Makine Öğrenmesi tabanlı tahmin
- İstatistiksel Analiz
- Bahis oranları bazlı hesaplama
""")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('attached_assets/oranlar1234.csv', low_memory=False)

        # Convert score columns to numeric
        score_columns = ['FTHG', 'FTAG', 'HTHG', 'HTAG']
        for col in score_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert odds columns to float
        odds_columns = [col for col in df.columns if any(x in col.upper() for x in ['ODD', 'PROB', 'RATE'])]
        for col in odds_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill NaN values with 0 for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        # Ensure team columns are strings
        df['HomeTeam'] = df['HomeTeam'].astype(str)
        df['AwayTeam'] = df['AwayTeam'].astype(str)

        st.session_state.df_loaded = True
        return df
    except Exception as e:
        st.error(f"Veri yükleme hatası: {str(e)}")
        st.write("Detaylı hata bilgisi:", e)
        return None

try:
    df = load_data()
    if df is not None:
        teams = sorted(df['HomeTeam'].unique())
        st.success("Veri başarıyla yüklendi!")
        st.write(f"Toplam maç sayısı: {len(df)}")
        st.write(f"Takım sayısı: {len(teams)}")

        # Strategy Advisor'ı başlat
        if st.session_state.strategy_advisor is None:
            st.session_state.strategy_advisor = StrategyAdvisor(df)
    else:
        st.error("Veri seti yüklenemedi. Lütfen dosyanın mevcut ve erişilebilir olduğunu kontrol edin.")
        st.stop()
except Exception as e:
    st.error(f"Veri işleme hatası: {str(e)}")
    st.write("Detaylı hata bilgisi:", e)
    st.stop()

# Team selection
col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("Ev Sahibi Takım", teams)

with col2:
    away_teams = [team for team in teams if team != home_team]
    away_team = st.selectbox("Deplasman Takımı", away_teams)

# Odds input
st.subheader("Bahis Oranları (Opsiyonel)")
odds_col1, odds_col2, odds_col3 = st.columns(3)

with odds_col1:
    home_odds = st.number_input("Ev Sahibi Galibiyet", min_value=1.0, step=0.01)

with odds_col2:
    draw_odds = st.number_input("Beraberlik", min_value=1.0, step=0.01)

with odds_col3:
    away_odds = st.number_input("Deplasman Galibiyet", min_value=1.0, step=0.01)

# Analysis button
analyze_button = st.button("Analizi Başlat", type="primary")

# Make predictions when teams are selected and button is clicked
if home_team and away_team and analyze_button:
    try:
        # Train models if not already trained
        if not st.session_state.models_trained:
            with st.spinner("Tahmin modelleri eğitiliyor..."):
                st.session_state.prediction_model.train(df)
                st.session_state.models_trained = True

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
        st.subheader("Maç Sonuç Tahmini")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.plotly_chart(create_probability_chart(
                home_team, away_team, ml_pred, "Makine Öğrenmesi"
            ), use_container_width=True)

        with col2:
            st.plotly_chart(create_probability_chart(
                home_team, away_team, stat_pred, "İstatistiksel"
            ), use_container_width=True)

        with col3:
            if odds_pred is not None:
                st.plotly_chart(create_probability_chart(
                    home_team, away_team, odds_pred, "Oran Bazlı"
                ), use_container_width=True)

        # Display final combined prediction
        st.subheader("Birleşik Tahmin")
        st.plotly_chart(create_probability_chart(
            home_team, away_team, final_pred, "Birleşik Model"
        ), use_container_width=True)

        # Goal prediction
        st.subheader("Gol Tahmini")
        goal_pred = st.session_state.prediction_model.predict_goals(features)
        st.write(f"Tahmini gol sayısı: {goal_pred:.1f}")

        # Over/Under probabilities
        over_under = st.session_state.prediction_model.predict_over_under(features)
        st.write(f"2.5 Üst olma olasılığı: {over_under[0]:.1%}")
        st.write(f"2.5 Alt olma olasılığı: {over_under[1]:.1%}")

        # Display team form comparison
        st.subheader("Takım Form Karşılaştırması")
        home_form = st.session_state.prediction_model.get_team_form(df, home_team)
        away_form = st.session_state.prediction_model.get_team_form(df, away_team)

        st.plotly_chart(create_form_chart(
            home_form,
            away_form,
            home_team, away_team
        ), use_container_width=True)

        # Display historical matches
        st.subheader("Son Karşılaşmalar")
        st.markdown(create_history_table(df, home_team, away_team), unsafe_allow_html=True)

        # Stratejik analiz bölümü
        st.subheader("Stratejik Analiz")

        # Takım stilleri
        col1, col2 = st.columns(2)
        with col1:
            home_style = st.session_state.strategy_advisor.analyze_team_style(home_team)
            st.write(f"**{home_team} Oyun Stili:**")
            for metric, value in home_style.items():
                st.progress(value, text=metric.replace('_', ' ').title())

        with col2:
            away_style = st.session_state.strategy_advisor.analyze_team_style(away_team)
            st.write(f"**{away_team} Oyun Stili:**")
            for metric, value in away_style.items():
                st.progress(value, text=metric.replace('_', ' ').title())

        # Taktik önerileri
        st.subheader("Taktik Önerileri")
        advice = st.session_state.strategy_advisor.generate_tactical_advice(home_team, away_team)

        tab1, tab2, tab3 = st.tabs(["Hücum", "Savunma", "Genel"])

        with tab1:
            for tip in advice['attacking']:
                st.info(tip)

        with tab2:
            for tip in advice['defensive']:
                st.warning(tip)

        with tab3:
            for tip in advice['general']:
                st.success(tip)

        # Kilit oyuncu rolleri
        st.subheader("Önemli Oyuncu Rolleri")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**{home_team} için önemli roller:**")
            for role in st.session_state.strategy_advisor.get_key_player_roles(home_team):
                st.write(f"• {role}")

        with col2:
            st.write(f"**{away_team} için önemli roller:**")
            for role in st.session_state.strategy_advisor.get_key_player_roles(away_team):
                st.write(f"• {role}")

        # Maç tempo tahmini
        st.subheader("Maç Tempo Tahmini")
        tempo = st.session_state.strategy_advisor.predict_match_tempo(home_team, away_team)
        st.info(tempo)

    except Exception as e:
        st.error(f"Tahmin hatası: {str(e)}")
        st.write("Detaylı hata bilgisi:", e)

# Footer
st.markdown("---")
st.markdown("Streamlit ile geliştirilmiştir • Veriler API-Football'dan alınmıştır")