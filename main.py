import streamlit as st
import pandas as pd
import numpy as np
from data_handler import DataHandler
from models import StatisticalModel, OddsBasedModel
from utils import (create_probability_chart, create_form_chart,
                  create_history_table, calculate_combined_prediction)
from strategy_advisor import StrategyAdvisor
from typing import Dict
import plotly.graph_objects as go

# Initialize session state
if 'data_handler' not in st.session_state:
    st.session_state.data_handler = DataHandler("c417cb5967msh54c12f850f3798cp12a733jsn315fab53ee3c")

if 'models' not in st.session_state:
    st.session_state.statistical_model = StatisticalModel()
    st.session_state.odds_model = OddsBasedModel()
    st.session_state.strategy_advisor = None

# Page config
st.set_page_config(page_title="Futbol Maç Tahmini", layout="wide")

# Title and description
st.title("Futbol Maç Tahmin Sistemi")
st.markdown("""
Bu sistem çoklu analiz yöntemleri kullanarak futbol maç sonuçlarını tahmin eder:
- İstatistiksel Analiz
- Bahis oranları bazlı hesaplama
- Güvenilirlik Simülasyonu
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
        # Get predictions from each model
        stat_pred = st.session_state.statistical_model.calculate_probabilities(df, home_team, away_team)

        odds_pred = None
        if all([home_odds > 1.0, draw_odds > 1.0, away_odds > 1.0]):
            odds_pred = st.session_state.odds_model.calculate_probabilities(home_odds, draw_odds, away_odds)

        # Calculate combined prediction (statistical and odds-based)
        if odds_pred is not None:
            final_pred = (stat_pred * 0.7 + odds_pred * 0.3)  # Give more weight to statistical model
        else:
            final_pred = stat_pred

        # Display predictions
        st.subheader("Maç Sonuç Tahmini")
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(create_probability_chart(
                home_team, away_team, stat_pred, "İstatistiksel"
            ), use_container_width=True)

        with col2:
            if odds_pred is not None:
                st.plotly_chart(create_probability_chart(
                    home_team, away_team, odds_pred, "Oran Bazlı"
                ), use_container_width=True)

        # Display final combined prediction
        st.subheader("Birleşik Tahmin")
        st.plotly_chart(create_probability_chart(
            home_team, away_team, final_pred, "Birleşik Model"
        ), use_container_width=True)

        # Tahmin güvenilirlik analizi
        st.subheader("Tahmin Güvenilirlik Analizi")

        reliability_analysis = st.session_state.strategy_advisor.analyze_prediction_reliability(
            home_team, away_team, final_pred
        )

        # Güvenilirlik skoru için progress bar
        reliability_score = reliability_analysis.get('reliability_score', 0.5)
        st.progress(reliability_score, text=f"Güvenilirlik Skoru: {reliability_score:.2%}")

        # Güven ve risk faktörleri için kolonlar
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Güven Faktörleri**")
            for factor in reliability_analysis.get('confidence_factors', []):
                st.success(f"✓ {factor}")

        with col2:
            st.write("**Risk Faktörleri**")
            for factor in reliability_analysis.get('risk_factors', []):
                st.warning(f"⚠ {factor}")

        # Genel tavsiye
        st.info(f"💡 **Tavsiye:** {reliability_analysis.get('recommendation', 'Analiz yapılamadı.')}")


        # Goal prediction section
        st.subheader("⚽ Gol Tahminleri")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Maç Sonu Gol Tahminleri**")
            expected_goals, over_under_probs = st.session_state.statistical_model.predict_goals(df, home_team, away_team)

            st.write(f"Tahmini toplam gol: {expected_goals:.2f}")

            goals_df = pd.DataFrame({
                'Bahis': ['0.5 Üst', '1.5 Üst', '2.5 Üst', '3.5 Üst'],
                'Olasılık': [f"{prob:.1%}" for prob in over_under_probs]
            })
            st.table(goals_df)

            # Karşılıklı gol tahmini
            btts_prob = st.session_state.statistical_model.predict_both_teams_to_score(df, home_team, away_team)
            st.metric("Karşılıklı Gol Olasılığı", f"{btts_prob:.1%}")

        with col2:
            st.markdown("**İlk Yarı Gol Tahminleri**")
            first_half_goals, first_half_probs = st.session_state.statistical_model.predict_first_half_goals(df, home_team, away_team)

            st.write(f"Tahmini ilk yarı gol: {first_half_goals:.2f}")

            first_half_df = pd.DataFrame({
                'Bahis': ['İY 0.5 Üst', 'İY 1.5 Üst'],
                'Olasılık': [f"{prob:.1%}" for prob in first_half_probs]
            })
            st.table(first_half_df)

        # Kart ve Korner tahminleri
        st.subheader("📊 Kart ve Korner Tahminleri")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Kart Tahminleri**")
            card_predictions = st.session_state.statistical_model.predict_cards(df, home_team, away_team)

            st.metric("Tahmini Toplam Kart", f"{card_predictions['expected_total']:.1f}")
            st.write(f"3.5 Alt Olasılığı: {card_predictions['under_3.5_cards']:.1%}")
            st.write(f"3.5 Üst Olasılığı: {card_predictions['over_3.5_cards']:.1%}")

        with col2:
            st.markdown("**Korner Tahminleri**")
            corner_predictions = st.session_state.statistical_model.predict_corners(df, home_team, away_team)

            st.metric("Tahmini Toplam Korner", f"{corner_predictions['expected_total']:.1f}")
            st.write(f"9.5 Alt Olasılığı: {corner_predictions['under_9.5_corners']:.1%}")
            st.write(f"9.5 Üst Olasılığı: {corner_predictions['over_9.5_corners']:.1%}")

        # Bahis önerisi bölümünü güncelle
        st.markdown("### 💰 Bahis Önerileri")
        betting_advice = st.session_state.strategy_advisor.generate_betting_advice(home_form, away_form)

        # Güven skoru göstergesi
        confidence = betting_advice['confidence_score']
        st.progress(confidence, text=f"Güven Skoru: {confidence:.1%}")

        # Önerileri düzenli göster
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Maç Sonucu**")
            outcome_probs = final_pred
            best_outcome_idx = np.argmax(outcome_probs)
            outcomes = ['1', 'X', '2']

            if confidence > 0.6:
                st.success(f"✅ Önerilen: {outcomes[best_outcome_idx]} ({outcome_probs[best_outcome_idx]:.1%})")
            else:
                st.info("⚠️ Yeterince güvenilir tahmin yok")

        with col2:
            st.markdown("**Gol Bahisleri**")
            if expected_goals > 2.5 and over_under_probs[2] > 0.6:
                st.success("✅ 2.5 Üst")
            elif expected_goals < 2 and (1 - over_under_probs[2]) > 0.6:
                st.success("✅ 2.5 Alt")

            if btts_prob > 0.65:
                st.success("✅ KG Var")
            elif btts_prob < 0.35:
                st.success("✅ KG Yok")

        with col3:
            st.markdown("**Diğer**")
            if first_half_goals > 1.2 and first_half_probs[1] > 0.55:
                st.success("✅ İY 1.5 Üst")
            elif first_half_goals < 0.8 and (1 - first_half_probs[0]) > 0.55:
                st.success("✅ İY 0.5 Alt")

            if card_predictions['expected_total'] > 4:
                st.success("✅ 3.5 Kart Üst")
            if corner_predictions['expected_total'] > 10.5:
                st.success("✅ 9.5 Korner Üst")

        # Risk faktörleri
        if betting_advice.get('risk_factors'):
            st.markdown("**⚠️ Risk Faktörleri:**")
            for factor in betting_advice['risk_factors']:
                st.warning(factor)

        # Display team form comparison
        st.subheader("Takım Form Karşılaştırması")
        try:
            home_form = st.session_state.strategy_advisor.get_team_form(df, home_team)
            away_form = st.session_state.strategy_advisor.get_team_form(df, away_team)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**{home_team} Son Form**")
                st.metric("Galibiyet", home_form['wins'])
                st.metric("Beraberlik", home_form['draws'])
                st.metric("Mağlubiyet", home_form['losses'])
                st.metric("Form Skoru", f"{home_form['form_score']:.2%}")
                st.metric("Seri", home_form['current_streak'])
                st.metric("Rakip Güç Ortalaması", f"{home_form['average_opponent_strength']:.2f}")

            with col2:
                st.markdown(f"**{away_team} Son Form**")
                st.metric("Galibiyet", away_form['wins'])
                st.metric("Beraberlik", away_form['draws'])
                st.metric("Mağlubiyet", away_form['losses'])
                st.metric("Form Skoru", f"{away_form['form_score']:.2%}")
                st.metric("Seri", away_form['current_streak'])
                st.metric("Rakip Güç Ortalaması", f"{away_form['average_opponent_strength']:.2f}")

            # Form karşılaştırma grafiği
            st.markdown("### Form Trendi")

            def create_form_chart(home_form: Dict, away_form: Dict, home_team: str, away_team: str):
                """Form karşılaştırma grafiği oluştur"""
                categories = ['Galibiyet', 'Beraberlik', 'Mağlubiyet', 'Gol Ortalaması', 'Form Skoru']

                home_values = [
                    home_form['wins'],
                    home_form['draws'],
                    home_form['losses'],
                    home_form['avg_goals_scored'],
                    home_form['form_score']
                ]

                away_values = [
                    away_form['wins'],
                    away_form['draws'],
                    away_form['losses'],
                    away_form['avg_goals_scored'],
                    away_form['form_score']
                ]

                fig = go.Figure()

                fig.add_trace(go.Scatterpolar(
                    r=home_values,
                    theta=categories,
                    fill='toself',
                    name=home_team
                ))

                fig.add_trace(go.Scatterpolar(
                    r=away_values,
                    theta=categories,
                    fill='toself',
                    name=away_team
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(max(home_values), max(away_values)) * 1.2]
                        )),
                    showlegend=True,
                    title="Form Karşılaştırması"
                )

                return fig

            fig = create_form_chart(home_form, away_form, home_team, away_team)
            st.plotly_chart(fig, use_container_width=True)

            # Bahis önerisi
            st.markdown("### 💰 Bahis Önerisi")
            betting_advice = st.session_state.strategy_advisor.generate_betting_advice(home_form, away_form)

            # Güven skoru göstergesi
            st.progress(betting_advice['confidence_score'],
                       text=f"Güven Skoru: {betting_advice['confidence_score']:.1%}")

            # Öneriler
            if betting_advice['recommendations']:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Önerilen Bahisler:**")
                    for rec in betting_advice['recommendations']:
                        if betting_advice['confidence_score'] > 0.7:
                            st.success(f"✅ {rec}")
                        else:
                            st.info(f"ℹ️ {rec}")

                with col2:
                    st.markdown("**Gerekçeler:**")
                    for exp in betting_advice['explanations']:
                        st.write(f"• {exp}")

                if betting_advice['confidence_score'] > 0.8:
                    st.success("🎯 Yüksek güvenilirlikli tahmin")
                elif betting_advice['confidence_score'] > 0.6:
                    st.info("📊 Orta güvenilirlikli tahmin")
                else:
                    st.warning("⚠️ Düşük güvenilirlikli tahmin")

        except Exception as e:
            st.error(f"Form karşılaştırması yapılırken hata oluştu: {str(e)}")
            st.write("Detaylı hata bilgisi:", e)

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