import streamlit as st
import pandas as pd
import numpy as np
import logging
from data_handler import DataHandler
from models import StatisticalModel, OddsBasedModel
from utils import (create_probability_chart, create_form_chart,
                  create_history_table, calculate_combined_prediction)
from strategy_advisor import StrategyAdvisor
from ml_predictor import MLPredictor
from typing import Dict, List
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
def initialize_session_state():
    """Oturum durumunu başlat"""
    if 'data_handler' not in st.session_state:
        st.session_state.data_handler = DataHandler()
    
    if 'ml_predictor' not in st.session_state:
        predictor = MLPredictor()
        # Modeli eğit
        X, y = predictor.create_training_data(num_samples=100)
        predictor.train_models((X, y))
        st.session_state.ml_predictor = predictor

    if 'strategy_advisor' not in st.session_state:
        st.session_state.strategy_advisor = None

    if 'statistical_model' not in st.session_state:
        st.session_state.statistical_model = StatisticalModel()

    if 'odds_model' not in st.session_state:
        st.session_state.odds_model = OddsBasedModel()

def make_prediction(match_id: int) -> Dict:
    """Maç tahmini yap"""
    try:
        # Maç verilerini al
        match_stats = st.session_state.data_handler.get_match_statistics(match_id)
        events = st.session_state.data_handler.get_match_events(match_id)
        historical_data = st.session_state.data_handler.get_historical_data(match_id)
        
        # Tahmin yap
        prediction = st.session_state.ml_predictor.predict_goals(
            match_stats, events, historical_data
        )
        
        return {
            'success': True,
            'prediction': prediction,
            'match_stats': match_stats,
            'events': events
        }
        
    except Exception as e:
        st.error(f"Tahmin yapılırken hata oluştu: {str(e)}")
        return {'success': False, 'error': str(e)}

def display_prediction_results(results: Dict):
    """Tahmin sonuçlarını göster"""
    if results['success']:
        prediction = results['prediction']
        match_stats = results['match_stats']
        
        st.subheader("Tahmin Sonuçları")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Tahmini Gol Sayısı", f"{prediction['predicted_goals']:.1f}")
            st.metric("Güven Skoru", f"{prediction['ensemble_confidence']:.1%}")
        
        with col2:
            # Maç istatistiklerini göster
            home_team = match_stats[0]['team']['name']
            away_team = match_stats[1]['team']['name']
            
            st.write(f"**{home_team}**")
            for stat in match_stats[0]['statistics']:
                if stat['type'] in ['Total Shots', 'Corner Kicks', 'Yellow Cards']:
                    st.write(f"{stat['type']}: {stat['value']}")
            
            st.write(f"\n**{away_team}**")
            for stat in match_stats[1]['statistics']:
                if stat['type'] in ['Total Shots', 'Corner Kicks', 'Yellow Cards']:
                    st.write(f"{stat['type']}: {stat['value']}")
    else:
        st.error(f"Hata: {results['error']}")

def calculate_first_half_goals(match_stats, historical_data):
    """İlk yarı gol tahminlerini hesapla"""
    try:
        # Temel istatistikler
        home_stats = match_stats[0]['statistics']
        away_stats = match_stats[1]['statistics']

        # Şut istatistikleri
        home_shots = int(home_stats[2]['value'] or 0)  # Toplam şut
        away_shots = int(away_stats[2]['value'] or 0)
        home_shots_on_target = int(home_stats[0]['value'] or 0)  # İsabetli şut
        away_shots_on_target = int(away_stats[0]['value'] or 0)
        
        # Hücum etkinliği
        home_dangerous_attacks = int(home_stats[13]['value'] or 0)
        away_dangerous_attacks = int(away_stats[13]['value'] or 0)
        home_possession = float(home_stats[9]['value'].strip('%') or 0)
        away_possession = float(away_stats[9]['value'].strip('%') or 0)

        # Tarihsel verilerden ilk yarı gol ortalaması ve form analizi
        historical_first_half_goals = 0
        home_form_score = 0
        away_form_score = 0
        if historical_data and 'recent_matches' in historical_data:
            matches = historical_data['recent_matches']
            total_first_half_goals = 0
            home_goals = 0
            away_goals = 0
            match_count = len(matches)
            
            for i, match in enumerate(matches):
                weight = 1 - (i / match_count)  # Son maçlara daha fazla ağırlık ver
                total_first_half_goals += match.get('first_half_goals', 1.2) * weight
                home_goals += match.get('home_goals', 0) * weight
                away_goals += match.get('away_goals', 0) * weight
            
            historical_first_half_goals = total_first_half_goals / match_count
            home_form_score = home_goals / match_count
            away_form_score = away_goals / match_count

        # Şut kalitesi skoru (0-1 arası)
        home_shot_quality = home_shots_on_target / max(home_shots, 1)
        away_shot_quality = away_shots_on_target / max(away_shots, 1)

        # Hücum etkinliği skoru (0-1 arası)
        home_attack_efficiency = (home_dangerous_attacks * 0.6 + home_possession * 0.4) / 100
        away_attack_efficiency = (away_dangerous_attacks * 0.6 + away_possession * 0.4) / 100

        # Maç temposunu hesapla (0-1 arası)
        total_shots = home_shots + away_shots
        total_dangerous_attacks = home_dangerous_attacks + away_dangerous_attacks
        match_intensity = (
            (total_shots / 20) * 0.3 +  # 20 şut normal bir maç için referans
            (total_dangerous_attacks / 30) * 0.4 +  # 30 tehlikeli atak normal bir maç için referans
            ((home_possession + away_possession) / 100) * 0.3
        )
        match_intensity = min(max(match_intensity, 0), 1)

        # Form bazlı düzeltme faktörü
        form_factor = (home_form_score + away_form_score) / 2

        # İlk yarı gol tahmini bileşenleri
        base_prediction = 1.2  # Ortalama ilk yarı gol sayısı
        shot_based_prediction = (home_shot_quality + away_shot_quality) * 1.5
        attack_based_prediction = (home_attack_efficiency + away_attack_efficiency) * 1.8
        historical_factor = historical_first_half_goals * 0.8

        # Ağırlıklı tahmin
        predicted_goals = (
            base_prediction * 0.2 +
            shot_based_prediction * 0.25 +
            attack_based_prediction * 0.25 +
            historical_factor * 0.2 +
            form_factor * 0.1
        ) * (0.8 + match_intensity * 0.4)  # Maç temposu bazlı düzeltme

        # Alt/üst olasılıkları için gelişmiş sigmoid fonksiyonu
        def calculate_over_probability(expected_goals, threshold, k=2.5):
            x0 = threshold  # Eşik değeri
            return 1 / (1 + np.exp(-k * (expected_goals - x0)))

        # Farklı gol barajları için olasılıklar
        over_probabilities = {
            '0.5': calculate_over_probability(predicted_goals, 0.5),
            '1.5': calculate_over_probability(predicted_goals, 1.5),
            '2.5': calculate_over_probability(predicted_goals, 2.5)
        }

        # Güven skoru hesaplama
        confidence_factors = [
            match_intensity,
            (home_shot_quality + away_shot_quality) / 2,
            (home_attack_efficiency + away_attack_efficiency) / 2,
            min(form_factor / 2, 1)
        ]
        confidence = sum(confidence_factors) / len(confidence_factors)
        confidence = min(max(confidence, 0.5), 0.95)

        return {
            'expected_goals': predicted_goals,
            'over_probabilities': over_probabilities,
            'confidence': confidence,
            'match_intensity': match_intensity,
            'shot_quality': (home_shot_quality + away_shot_quality) / 2,
            'attack_efficiency': (home_attack_efficiency + away_attack_efficiency) / 2,
            'form_impact': form_factor
        }

    except Exception as e:
        logger.error(f"İlk yarı gol tahmini hesaplanırken hata: {str(e)}")
        return {
            'expected_goals': 1.2,
            'over_probabilities': {'0.5': 0.5, '1.5': 0.5, '2.5': 0.3},
            'confidence': 0.5,
            'match_intensity': 0.5,
            'shot_quality': 0.5,
            'attack_efficiency': 0.5,
            'form_impact': 0.5
        }

def display_live_matches(matches: List[Dict]):
    """Canlı maçları göster"""
    for match in matches:
        stats_available = match.get('statistics_available', False)
        
        st.write(f"{match['home_team']} {match['score']} {match['away_team']} ({match['minute']}')")
        
        if not stats_available:
            st.info("Maç istatistikleri şu an için mevcut değil. Tahminler sınırlı veriyle oluşturuluyor.")
        else:
            # İstatistikleri göster
            display_match_stats(match['statistics'])

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

# Sekme seçimi
tab_main, tab_live, tab_test = st.tabs(["Ana Sayfa", "Canlı Maçlar", "Test Tahmin"])

with tab_main:
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

            # Initialize session state
            initialize_session_state()

            # Strategy Advisor'ı başlat
            if st.session_state.strategy_advisor is None:
                from strategy_advisor import StrategyAdvisor
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


            try:
                # Form verilerini al
                home_form = st.session_state.strategy_advisor.get_team_form(df, home_team)
                away_form = st.session_state.strategy_advisor.get_team_form(df, away_team)

                # Gol tahminlerini hesapla
                expected_goals, over_under_probs = st.session_state.statistical_model.predict_goals(df, home_team, away_team)
                btts_prob = st.session_state.statistical_model.predict_both_teams_to_score(df, home_team, away_team)
                first_half_predictions = calculate_first_half_goals(st.session_state.data_handler.get_match_statistics(home_team=home_team, away_team=away_team), st.session_state.data_handler.get_historical_data(home_team=home_team, away_team=away_team))
                card_predictions = st.session_state.statistical_model.predict_cards(df, home_team, away_team)
                corner_predictions = st.session_state.statistical_model.predict_corners(df, home_team, away_team)

                # Display team form comparison
                st.subheader("Takım Form Karşılaştırması")
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
                fig = create_form_chart(home_form, away_form, home_team, away_team)
                st.plotly_chart(fig, use_container_width=True)

                # Bahis önerisi bölümü
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
                    if first_half_predictions['expected_goals'] > 1.2 and first_half_predictions['over_1.5_prob'] > 0.55:
                        st.success("✅ İY 1.5 Üst")
                    elif first_half_predictions['expected_goals'] < 0.8 and (1 - first_half_predictions['under_1.5_prob']) > 0.55:
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

                # Goal prediction section
                st.subheader("⚽ Gol Tahminleri")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Maç Sonu Gol Tahminleri**")
                    st.write(f"Tahmini toplam gol: {expected_goals:.2f}")

                    goals_df = pd.DataFrame({
                        'Bahis': ['0.5 Üst', '1.5 Üst', '2.5 Üst', '3.5 Üst'],
                        'Olasılık': [f"{prob:.1%}" for prob in over_under_probs]
                    })
                    st.table(goals_df)
                    st.metric("Karşılıklı Gol Olasılığı", f"{btts_prob:.1%}")

                with col2:
                    st.markdown("**İlk Yarı Gol Tahminleri**")
                    st.write(f"Tahmini ilk yarı gol: {first_half_predictions['expected_goals']:.2f}")

                    # İlk yarı gol olasılıkları tablosu
                    first_half_df = pd.DataFrame({
                        'Bahis': ['İY 0.5 Üst', 'İY 1.5 Üst', 'İY 2.5 Üst'],
                        'Olasılık': [
                            f"{first_half_predictions['over_probabilities']['0.5']:.1%}",
                            f"{first_half_predictions['over_probabilities']['1.5']:.1%}",
                            f"{first_half_predictions['over_probabilities']['2.5']:.1%}"
                        ]
                    })
                    st.table(first_half_df)

                    # Güven skoru ve maç temposu
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Güven Skoru", f"{first_half_predictions['confidence']:.1%}")
                    with col2:
                        st.metric("Maç Temposu", f"{first_half_predictions['match_intensity']:.1%}")

                # Kart ve Korner tahminleri
                st.subheader("📊 Kart ve Korner Tahminleri")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Kart Tahminleri**")
                    st.metric("Tahmini Toplam Kart", f"{card_predictions['expected_total']:.1f}")
                    st.write(f"3.5 Alt Olasılığı: {card_predictions['under_3.5_cards']:.1%}")
                    st.write(f"3.5 Üst Olasılığı: {card_predictions['over_3.5_cards']:.1%}")

                with col2:
                    st.markdown("**Korner Tahminleri**")
                    st.metric("Tahmini Toplam Korner", f"{corner_predictions['expected_total']:.1f}")
                    st.write(f"9.5 Alt Olasılığı: {corner_predictions['under_9.5_corners']:.1%}")
                    st.write(f"9.5 Üst Olasılığı: {corner_predictions['over_9.5_corners']:.1%}")

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

with tab_live:
    st.subheader("Canlı Maçlar")
    
    if st.button("Canlı Maçları Yenile"):
        try:
            # DataHandler'ı başlat
            initialize_session_state()
            
            # Canlı maçları al
            live_matches = st.session_state.data_handler.get_live_matches()
            
            if live_matches:
                for match in live_matches:
                    # Maç bilgilerini al
                    home_team = match.get('teams', {}).get('home', {}).get('name', '')
                    away_team = match.get('teams', {}).get('away', {}).get('name', '')
                    score = match.get('goals', {})
                    home_score = score.get('home', 0)
                    away_score = score.get('away', 0)
                    match_time = match.get('fixture', {}).get('status', {}).get('elapsed', 0)
                    match_id = match.get('fixture', {}).get('id')
                    
                    # Maç kartını oluştur
                    with st.container():
                        st.markdown(f"""
                        ### {home_team} {home_score} - {away_score} {away_team}
                        **Dakika:** {match_time}'
                        """)
                        
                        # Tahmin butonu
                        button_key = f"predict_button_{match_id}"
                        if st.button(f"Tahmin Yap ({home_team} vs {away_team})", key=button_key):
                            try:
                                # Maç verilerini al
                                match_stats = st.session_state.data_handler.get_match_statistics(match_id=match_id)
                                events = st.session_state.data_handler.get_match_events(home_team=home_team, away_team=away_team)
                                historical_data = st.session_state.data_handler.get_historical_data(home_team=home_team, away_team=away_team)

                                # Tahmin yap
                                prediction = st.session_state.ml_predictor.predict_goals(
                                    match_stats, events, historical_data
                                )

                                # Sonuçları göster
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Tahmini Gol Sayısı", f"{prediction['predicted_goals']:.2f}")
                                    st.metric("Model 1 Güven Skoru", f"{prediction['model1_confidence']:.1%}")
                                    st.metric("Model 2 Güven Skoru", f"{prediction['model2_confidence']:.1%}")
                                    st.metric("Ensemble Güven Skoru", f"{prediction['ensemble_confidence']:.1%}")

                                with col2:
                                    st.write("**Maç İstatistikleri**")
                                    st.write(f"**{home_team}:**")
                                    for stat in match_stats[0]['statistics']:
                                        if stat['type'] in ['Total Shots', 'Corner Kicks', 'Yellow Cards']:
                                            st.write(f"{stat['type']}: {stat['value']}")
                                    
                                    st.write(f"\n**{away_team}:**")
                                    for stat in match_stats[1]['statistics']:
                                        if stat['type'] in ['Total Shots', 'Corner Kicks', 'Yellow Cards']:
                                            st.write(f"{stat['type']}: {stat['value']}")

                            except Exception as e:
                                st.error(f"Tahmin yapılırken hata oluştu: {str(e)}")
                        
                        st.markdown("---")
            else:
                st.info("Şu anda canlı maç bulunmuyor.")
                
        except Exception as e:
            st.error(f"Canlı maçlar alınırken hata oluştu: {str(e)}")
            st.write("Detaylı hata:", e)

with tab_test:
    st.subheader("Test Tahmin")
    if st.button("Test Tahmini Yap"):
        try:
            # DataHandler ve MLPredictor'ı başlat
            initialize_session_state()

            # Örnek takım isimleri
            home_team = "Fenerbahçe"
            away_team = "Galatasaray"

            # Maç verilerini al
            match_stats = st.session_state.data_handler.get_match_statistics(home_team, away_team)
            events = st.session_state.data_handler.get_match_events(home_team, away_team)
            historical_data = st.session_state.data_handler.get_historical_data(home_team, away_team)

            # Tahmin yap
            prediction = st.session_state.ml_predictor.predict_goals(
                match_stats, events, historical_data
            )

            # Sonuçları göster
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Tahmini Gol Sayısı", f"{prediction['predicted_goals']:.2f}")
                st.metric("Model 1 Güven Skoru", f"{prediction['model1_confidence']:.1%}")
                st.metric("Model 2 Güven Skoru", f"{prediction['model2_confidence']:.1%}")
                st.metric("Ensemble Güven Skoru", f"{prediction['ensemble_confidence']:.1%}")

            with col2:
                st.write("**Maç İstatistikleri**")
                st.write(f"**{home_team}:**")
                for stat in match_stats[0]['statistics']:
                    if stat['type'] in ['Total Shots', 'Corner Kicks', 'Yellow Cards']:
                        st.write(f"{stat['type']}: {stat['value']}")
                
                st.write(f"\n**{away_team}:**")
                for stat in match_stats[1]['statistics']:
                    if stat['type'] in ['Total Shots', 'Corner Kicks', 'Yellow Cards']:
                        st.write(f"{stat['type']}: {stat['value']}")

            # Olayları göster
            st.write("\n**Maç Olayları**")
            for event in events:
                st.write(f"{event['time']['elapsed']}' - {event['team']['name']} - {event['type']}")

        except Exception as e:
            st.error(f"Test tahmini yapılırken hata oluştu: {str(e)}")
            st.write("Detaylı hata:", e)