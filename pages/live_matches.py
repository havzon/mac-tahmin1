import streamlit as st
import time
import logging
from typing import Dict
from data_handler import DataHandler
from models import StatisticalModel
from utils import create_probability_chart, create_form_chart
from strategy_advisor import StrategyAdvisor
from match_commentator import MatchCommentator
from performance_analyzer import PerformanceAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Canlı Maçlar", layout="wide")

# Initialize session state
if 'data_handler' not in st.session_state:
    try:
        api_key = st.secrets.get("RAPIDAPI_KEY")
        if api_key:
            st.session_state.data_handler = DataHandler(api_key)
            st.session_state.statistical_model = StatisticalModel()
            st.session_state.strategy_advisor = StrategyAdvisor(None)
            st.session_state.performance_analyzer = PerformanceAnalyzer()
            logger.info("All components initialized successfully")
        else:
            logger.error("API key not found in secrets")
            st.error("API anahtarı bulunamadı. Lütfen RAPIDAPI_KEY'i Secrets kısmına ekleyin.")
            st.stop()
    except Exception as e:
        logger.error(f"Error initializing API connection: {str(e)}")
        st.error(f"API bağlantısı sırasında hata oluştu: {str(e)}")
        st.stop()

# Initialize commentator separately to handle any potential errors
if 'commentator' not in st.session_state:
    try:
        logger.info("Initializing MatchCommentator")
        st.session_state.commentator = MatchCommentator()
    except Exception as e:
        logger.error(f"Error initializing commentator: {str(e)}")
        st.error(f"Yorumlayıcı başlatılırken hata oluştu: {str(e)}")
        st.session_state.commentator = None

def display_prediction_with_confidence(prediction: Dict):
    """Gol tahminini güven seviyeleriyle göster"""
    if not prediction:
        st.warning("Tahmin verileri bulunamadı.")
        return

    if 'error' in prediction:
        st.error(f"Tahmin hatası: {prediction['error']}")
        return

    st.subheader("Gol Tahmini Analizi")

    # Ana tahmin (en yüksek güvenli tahmin)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Önerilen Tahmin:**", prediction['prediction'])
        if prediction['expected_time']:
            st.write("**Tahmini Zaman:**", f"{prediction['expected_time']}. dakika")
        st.write("**Güven Seviyesi:**", prediction['confidence'].title())

    with col2:
        st.progress(prediction['probability'], text=f"Olasılık: {prediction['probability']:.1%}")

    # Bahis oranları gösterimi
    if 'betting_odds' in prediction:
        st.markdown("### Bahis Oranları Analizi")
        odds_col1, odds_col2 = st.columns(2)

        with odds_col1:
            st.markdown("**Canlı Bahis Oranları**")
            for bet_type, odd in prediction['betting_odds'].items():
                st.markdown(f"- {bet_type.replace('_', ' ').title()}: **{odd}**")

        with odds_col2:
            if 'implied_probabilities' in prediction:
                st.markdown("**İma Edilen Olasılıklar**")
                for bet_type, prob in prediction['implied_probabilities'].items():
                    st.markdown(f"- {bet_type.replace('_', ' ').title()}: **{prob:.1%}**")

    # Tüm güven seviyelerindeki tahminler
    if 'predictions' in prediction:
        st.markdown("### Farklı Güven Seviyeli Tahminler")

        # Güven seviyeleri için sekmeler
        tabs = st.tabs(["Yüksek Güven", "Orta Güven", "Düşük Güven"])
        confidence_colors = {
            'yüksek': '#2ecc71',  # Yeşil
            'orta': '#f1c40f',    # Sarı
            'düşük': '#e74c3c'    # Kırmızı
        }

        for tab, (level, color) in zip(tabs, confidence_colors.items()):
            with tab:
                if level in prediction['predictions']:
                    pred = prediction['predictions'][level]
                    st.markdown(f"### {level.title()} Güvenli Tahmin")

                    # Tahmin detayları
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Tahmin Detayları**")
                        st.markdown(f"Olasılık: **{pred['probability']:.1%}**")
                        if 'reason' in pred:
                            st.markdown(f"**Tahmin Nedenleri:**")
                            for reason in pred['reason'].split(" & "):
                                st.markdown(f"- *{reason}*")

                    with col2:
                        if 'probability' in pred:
                            import plotly.graph_objects as go
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = pred['probability'] * 100,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': f"{level.title()} Güven"},
                                gauge = {
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': color},
                                    'steps': [
                                        {'range': [0, 33], 'color': 'lightgray'},
                                        {'range': [33, 66], 'color': 'gray'},
                                        {'range': [66, 100], 'color': 'darkgray'}
                                    ]
                                }
                            ))
                            fig.update_layout(height=250)
                            st.plotly_chart(fig, use_container_width=True)

                    # Kalite faktörleri
                    if 'quality_factors' in pred:
                        st.markdown("**Kalite Faktörleri**")
                        quality_factors_data = []
                        for factor, value in pred['quality_factors'].items():
                            quality_factors_data.append({
                                'Factor': factor.replace('_', ' ').title(),
                                'Value': value * 100
                            })

                        if quality_factors_data:
                            import plotly.express as px
                            fig = px.bar(quality_factors_data,
                                       x='Factor',
                                       y='Value',
                                       text=[f'{v:.1f}%' for v in [d['Value'] for d in quality_factors_data]],
                                       title="Tahmin Kalite Faktörleri",
                                       height=300)
                            fig.update_layout(yaxis_range=[0, 100])
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Bu güven seviyesinde tahmin bulunmuyor.")

    # Maç durumu ve momentum bilgileri
    if 'momentum' in prediction and 'match_state' in prediction:
        st.markdown("### Maç ve Momentum Analizi")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Maç Durumu**")
            if 'phase' in prediction['match_state']:
                st.markdown(f"**Faz:** {prediction['match_state']['phase'].replace('_', ' ').title()}")
            if 'intensity' in prediction['match_state']:
                st.progress(prediction['match_state']['intensity'], 
                          text=f"Maç Yoğunluğu: {prediction['match_state']['intensity']:.1%}")

        with col2:
            st.markdown("**Momentum Analizi**")
            if 'trend' in prediction['momentum']:
                st.markdown(f"**Trend:** {prediction['momentum']['trend'].replace('_', ' ').title()}")
            if 'total' in prediction['momentum']:
                st.progress(min(1.0, prediction['momentum']['total']), 
                          text=f"Toplam Momentum: {min(1.0, prediction['momentum']['total']):.1%}")

def display_player_analysis(player_stats: Dict):
    """Oyuncu analiz sonuçlarını göster"""
    if not player_stats:
        st.warning("Oyuncu analizi için yeterli veri yok.")
        return

    st.subheader("Oyuncu Performans Analizi")

    # Oyuncuları performans skorlarına göre sırala
    sorted_players = sorted(
        player_stats.items(),
        key=lambda x: x[1]['rating'],
        reverse=True
    )

    # Her oyuncu için bir sekme oluştur
    player_tabs = st.tabs([f"{stats['name']} ({stats['team']})" for _, stats in sorted_players])

    for tab, (player_id, stats) in zip(player_tabs, sorted_players):
        with tab:
            st.markdown(f"### Puan: {stats['rating']:.1f}/10")
            st.markdown(f"**Performans Özeti:** {stats['summary']}")

            if stats.get('key_moments'):
                st.markdown("**Önemli Anlar**")
                for moment in stats['key_moments']:
                    st.info(f"{moment['time']}' - {moment['event_type']}")

def display_match_details(fixture_id, match_info):
    """Display detailed statistics and events for a match"""
    try:
        with st.spinner("Maç detayları yükleniyor..."):
            # Get match statistics
            stats = st.session_state.data_handler.get_match_statistics(fixture_id)
            events = st.session_state.data_handler.get_match_events(fixture_id)

            if not stats:
                logger.warning(f"No statistics available for fixture {fixture_id}")
                st.warning("Maç istatistikleri şu an için mevcut değil.")
                return

            # Score information
            score = [match_info['goals']['home'], match_info['goals']['away']]

            # Calculate live win probabilities
            win_probs = calculate_live_win_probability(stats, score)

            # Detaylı performans analizi
            try:
                if hasattr(st.session_state, 'performance_analyzer'):
                    performance_analysis = st.session_state.performance_analyzer.analyze_team_performance(stats, events)
                    player_analysis = st.session_state.performance_analyzer.analyze_player_performance(events)
                else:
                    logger.error("Performance analyzer not initialized")
                    performance_analysis = None
                    player_analysis = None
            except Exception as e:
                logger.error(f"Error in performance analysis: {str(e)}")
                performance_analysis = None
                player_analysis = None

            # AI Commentary Section
            if st.session_state.commentator is not None:
                st.subheader("Maç Yorumu")
                commentary = st.session_state.commentator.generate_match_commentary(stats, score, events)
                st.markdown(f"💬 {commentary}")

                # Next Goal Prediction
                next_goal = st.session_state.commentator.predict_next_goal(stats, events)
                display_prediction_with_confidence(next_goal)

            # Takım Performans Analizi
            if performance_analysis:
                st.markdown("---")
                st.header("Detaylı Performans Analizi")

                tab1, tab2, tab3 = st.tabs(["Ev Sahibi Analizi", "Deplasman Analizi", "Oyuncu Analizi"])

                with tab1:
                    display_team_analysis(performance_analysis['home_team'], 'home')

                with tab2:
                    display_team_analysis(performance_analysis['away_team'], 'away')

                with tab3:
                    display_player_analysis(player_analysis)

            # Display win probability chart
            st.subheader("Canlı Kazanma Olasılıkları")
            st.plotly_chart(create_probability_chart(
                match_info['teams']['home']['name'],
                match_info['teams']['away']['name'],
                win_probs,
                "Canlı Tahmin"
            ), use_container_width=True)

            # Display basic statistics
            if stats:
                st.subheader("Maç İstatistikleri")
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**{match_info['teams']['home']['name']}**")
                    for stat in stats[0]['statistics']:
                        if stat['value'] is not None:
                            st.metric(
                                label=stat['type'],
                                value=stat['value'],
                                delta=None
                            )

                with col2:
                    st.write(f"**{match_info['teams']['away']['name']}**")
                    for stat in stats[1]['statistics']:
                        if stat['value'] is not None:
                            st.metric(
                                label=stat['type'],
                                value=stat['value'],
                                delta=None
                            )

            # Display match events
            if events:
                st.subheader("Önemli Olaylar")
                for event in events:
                    if event['type'] == 'Goal':
                        st.success(format_event(event))
                    elif event['type'] in ['Card', 'subst']:
                        st.warning(format_event(event))
                    else:
                        st.info(format_event(event))

    except Exception as e:
        logger.error(f"Error displaying match details: {str(e)}")
        st.error(f"Maç detayları gösterilirken bir hata oluştu: {str(e)}")

def format_event(event):
    """Format match event for display"""
    try:
        return f"{event['time']['elapsed']}' - {event['team']['name']}: {event['type']} - {event['player']['name']}"
    except KeyError as e:
        logger.error(f"Error formatting event: {str(e)}")
        return "Event format error"

def calculate_live_win_probability(stats, score):
    """Calculate live win probability based on current stats and score"""
    if not stats:
        return [0.33, 0.34, 0.33]  # Default probabilities when no stats available

    home_score, away_score = score

    # Convert stats to numerical values for analysis
    try:
        home_stats = {
            'possession': float(stats[0]['statistics'][9]['value'].strip('%')) / 100 if stats[0]['statistics'][9]['value'] else 0.5,
            'shots_on_target': int(stats[0]['statistics'][2]['value'] or 0),
            'dangerous_attacks': int(stats[0]['statistics'][13]['value'] or 0)
        }

        away_stats = {
            'possession': float(stats[1]['statistics'][9]['value'].strip('%')) / 100 if stats[1]['statistics'][9]['value'] else 0.5,
            'shots_on_target': int(stats[1]['statistics'][2]['value'] or 0),
            'dangerous_attacks': int(stats[1]['statistics'][13]['value'] or 0)
        }
    except (IndexError, KeyError, ValueError) as e:
        logger.error(f"Error parsing match statistics: {str(e)}")
        return [0.33, 0.34, 0.33]

    # Calculate base probability from score
    score_factor = (home_score - away_score) * 0.15

    # Calculate stats-based probability
    stats_prob = (
        (home_stats['possession'] - 0.5) * 0.3 +
        (home_stats['shots_on_target'] - away_stats['shots_on_target']) * 0.05 +
        (home_stats['dangerous_attacks'] - away_stats['dangerous_attacks']) * 0.02
    )

    # Combine probabilities
    base_prob = 0.5  # Start with even probability
    final_prob = base_prob + score_factor + stats_prob

    # Normalize to valid probability range
    final_prob = max(0.1, min(0.9, final_prob))

    # Calculate draw and away win probabilities
    draw_prob = (1 - abs(score_factor)) * 0.3
    away_prob = 1 - final_prob - draw_prob

    return [final_prob, draw_prob, away_prob]

# Main page content
st.title("Canlı Maçlar")

try:
    # Auto refresh checkbox
    auto_refresh = st.checkbox("Otomatik Yenile (30 saniye)", value=True)

    # Get live matches
    with st.spinner("Canlı maçlar yükleniyor..."):
        live_matches = st.session_state.data_handler.get_live_matches()

    if live_matches:
        for match in live_matches:
            # Create an expander for each match
            with st.expander(f"{match['teams']['home']['name']} {match['goals']['home']} - {match['goals']['away']} {match['teams']['away']['name']} ({match['fixture']['status']['elapsed']}')"):
                display_match_details(match['fixture']['id'], match)
    else:
        st.info("Şu anda canlı maç bulunmuyor.")

    # Auto refresh logic
    if auto_refresh:
        time.sleep(30)
        st.experimental_rerun()

except Exception as e:
    logger.error(f"Error in main page execution: {str(e)}")
    st.error(f"Bir hata oluştu: {str(e)}")
    st.info("Sayfayı yenilemek için F5 tuşuna basın veya sayfayı manuel olarak yenileyin.")

def display_team_analysis(analysis: Dict, team_type: str):
    """Takım analiz sonuçlarını göster"""
    if not analysis or 'metrics' not in analysis:
        st.warning("Takım analizi için yeterli veri yok.")
        return

    metrics = analysis['metrics']
    st.subheader(f"{'Ev Sahibi' if team_type == 'home' else 'Deplasman'} Takım Analizi")

    # Performans skoru
    st.metric("Performans Skoru", f"{analysis['performance_score']:.2%}")

    # Metrikler
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Hücum Metrikleri**")
        st.progress(metrics['offensive_efficiency'], 
                   text=f"Hücum Etkinliği: {metrics['offensive_efficiency']:.2%}")
        st.progress(metrics['transition_speed'],
                   text=f"Geçiş Hızı: {metrics['transition_speed']:.2%}")

    with col2:
        st.write("**Savunma Metrikleri**")
        st.progress(metrics['defensive_stability'],
                   text=f"Savunma İstikrarı: {metrics['defensive_stability']:.2%}")
        st.progress(metrics['pressing_intensity'],
                   text=f"Pressing Yoğunluğu: {metrics['pressing_intensity']:.2%}")

    # Top kontrolü
    st.progress(metrics['possession_control'],
               text=f"Top Kontrolü: {metrics['possession_control']:.2%}")

    # Güçlü ve zayıf yönler
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Güçlü Yönler**")
        for strength in analysis.get('strengths', []):
            st.success(f"✓ {strength}")

    with col2:
        st.write("**Geliştirilmesi Gereken Alanlar**")
        for area in analysis.get('areas_to_improve', []):
            st.warning(f"⚠ {area}")

    # Performans trendi
    if 'trend' in analysis and 'period_scores' in analysis['trend']:
        st.write("**Performans Trendi**")
        for period, score in analysis['trend']['period_scores'].items():
            period_name = {
                'first_15': 'İlk 15 dakika',
                'mid_game': 'Orta bölüm',
                'last_15': 'Son 15 dakika'
            }[period]
            st.progress(score, text=f"{period_name}: {score:.2f}")