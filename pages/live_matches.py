import streamlit as st
import time
import logging
from typing import Dict
from data_handler import DataHandler
from models import StatisticalModel
from utils import create_probability_chart, create_form_chart
from strategy_advisor import StrategyAdvisor
from match_commentator import MatchCommentator
#from performance_analyzer import PerformanceAnalyzer #Removed as per intention
from simulation_engine import SimulationEngine #Added
import pandas as pd #Added
import plotly.express as px #Added


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
            #st.session_state.performance_analyzer = PerformanceAnalyzer() #Removed as per intention
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

if 'simulation_engine' not in st.session_state:
    st.session_state.simulation_engine = SimulationEngine()

def display_prediction_with_confidence(prediction: Dict):
    """Gelişmiş gol tahminini güven seviyeleriyle göster"""
    if not prediction:
        st.warning("Tahmin verileri bulunamadı.")
        return

    if 'error' in prediction:
        st.error(f"Tahmin hatası: {prediction['error']}")
        return

    st.subheader("Gol Tahmini Analizi")

    # Ana tahmin
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Önerilen Tahmin:**", prediction['prediction'])
        st.write("**Güven Seviyesi:**", prediction['confidence'].title())

    with col2:
        st.progress(prediction['probability'], text=f"Olasılık: {prediction['probability']:.1%}")

    # Bahis oranları analizi
    if 'betting_odds' in prediction and 'odds_analysis' in prediction:
        st.markdown("### Bahis Oranları Analizi")

        # Market kalitesi göstergesi
        market_quality = prediction['odds_analysis']['market_quality']
        st.metric(
            "Market Güven Skoru",
            f"{market_quality['confidence']:.1%}",
            delta=f"{(market_quality['efficiency'] - 0.5) * 100:.1f}%",
            delta_color="normal"
        )

        # Bahis oranları karşılaştırma
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Canlı Bahis Oranları")
            for bet_type, odds in prediction['betting_odds'].items():
                if bet_type in ['home', 'draw', 'away']:
                    category = "Maç Sonucu"
                elif bet_type in ['over25', 'under25', 'btts']:
                    category = "Gol Bahisleri"
                else:
                    category = "Diğer"

                st.metric(
                    f"{category} - {bet_type.replace('_', ' ').title()}", 
                    f"{odds:.2f}",
                    delta=f"{(1/float(odds) * 100):.1f}%",
                    delta_color="off"
                )

        with col2:
            st.markdown("#### İma Edilen Olasılıklar")
            for bet_type, prob in prediction['odds_analysis']['implied_probabilities'].items():
                st.metric(
                    bet_type.replace('_', ' ').title(),
                    f"{prob:.1%}",
                    delta=None
                )

        # Bahis pazarı trend analizi
        if 'odds_by_type' in prediction['odds_analysis']:
            st.markdown("#### Bahis Pazarı Trend Analizi")
            odds_by_type = prediction['odds_analysis']['odds_by_type']

            for category, odds in odds_by_type.items():
                if odds:  # Eğer kategori boş değilse
                    with st.expander(f"{category.replace('_', ' ').title()} Detayları"):
                        for bet_type, odd in odds.items():
                            implied_prob = 1/float(odd)
                            st.metric(
                                bet_type.replace('_', ' ').title(),
                                f"{odd:.2f}",
                                delta=f"{implied_prob:.1%}",
                                delta_color="off"
                            )

    # Tahmin detayları
    if 'predictions' in prediction:
        st.markdown("### Tahmin Detayları")
        tabs = st.tabs(["Yüksek Güven", "Düşük Güven"])

        for tab, level in zip(tabs, ['yüksek', 'düşük']):
            with tab:
                if level in prediction['predictions']:
                    pred = prediction['predictions'][level]
                    st.markdown(f"**Olasılık:** {pred['probability']:.1%}")
                    if 'reason' in pred:
                        st.markdown("**Tahmin Nedenleri:**")
                        for reason in pred['reason'].split(" & "):
                            st.markdown(f"- *{reason}*")
                else:
                    st.warning(f"Bu güven seviyesinde tahmin bulunmuyor.")



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

            # AI Commentary Section
            if st.session_state.commentator is not None:
                st.subheader("Maç Yorumu")
                commentary = st.session_state.commentator.generate_match_commentary(stats, score, events)
                st.markdown(f"💬 {commentary}")

                # Next Goal Prediction
                next_goal = st.session_state.commentator.predict_next_goal(stats, events)
                display_prediction_with_confidence(next_goal)

            # Display win probability chart
            st.subheader("Canlı Kazanma Olasılıkları")
            win_prob_chart = create_probability_chart(
                match_info['teams']['home']['name'],
                match_info['teams']['away']['name'],
                win_probs,
                "Canlı Tahmin"
            )
            st.plotly_chart(win_prob_chart, use_container_width=True, key=f"win_prob_{fixture_id}")


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

        # Simülasyon arayüzünü ekle
        st.markdown("---")
        display_simulation_interface(match_info, fixture_id)

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

def display_simulation_interface(match_info: Dict, fixture_id: str):
    """Tahmin simülasyonu arayüzünü göster"""
    st.markdown("### 🎮 Tahmin Simülasyonu")

    # Simülasyon ID'sini kontrol et veya oluştur
    sim_key = f"sim_{fixture_id}"
    if sim_key not in st.session_state:
        simulation_id = st.session_state.simulation_engine.create_simulation(
            fixture_id,
            {'match_info': match_info}
        )
        st.session_state[sim_key] = simulation_id

    simulation_id = st.session_state[sim_key]

    # Tahmin girişi
    with st.expander("Tahmin Yap", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            home_goals = st.number_input(
                f"{match_info['teams']['home']['name']} Gol",
                min_value=0,
                max_value=10,
                value=match_info['goals']['home'],
                key=f"home_goals_{fixture_id}"
            )

        with col2:
            away_goals = st.number_input(
                f"{match_info['teams']['away']['name']} Gol",
                min_value=0,
                max_value=10,
                value=match_info['goals']['away'],
                key=f"away_goals_{fixture_id}"
            )

        with col3:
            confidence = st.slider(
                "Tahmin Güveniniz",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                key=f"confidence_{fixture_id}"
            )

        # Tahmin türü seçimi
        prediction_type = st.selectbox(
            "Tahmin Türü",
            options=["Maç Sonucu", "Toplam Gol", "İlk Yarı Sonucu"],
            key=f"pred_type_{fixture_id}"
        )

        # Tahmin açıklaması
        prediction_notes = st.text_area(
            "Tahmin Notlarınız",
            placeholder="Tahmininizin nedenlerini açıklayın...",
            key=f"pred_notes_{fixture_id}"
        )

        if st.button("Tahmini Kaydet", key=f"save_pred_{fixture_id}"):
            prediction = {
                'home_goals': home_goals,
                'away_goals': away_goals,
                'confidence': confidence,
                'type': prediction_type,
                'notes': prediction_notes,
                'result': home_goals - away_goals  # Basit bir sonuç metriği
            }

            if st.session_state.simulation_engine.add_prediction(simulation_id, prediction):
                st.success("Tahmin başarıyla kaydedildi!")
            else:
                st.error("Tahmin kaydedilirken bir hata oluştu.")

    # Simülasyon istatistikleri
    stats = st.session_state.simulation_engine.get_simulation_stats(simulation_id)
    if stats:
        st.markdown("### 📊 Simülasyon İstatistikleri")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Toplam Tahmin", stats['total_predictions'])

        with col2:
            st.metric("Ortalama Güven", f"{stats['average_confidence']:.2%}")

        if stats['prediction_timeline']:
            # Tahmin zaman çizelgesi
            df = pd.DataFrame(stats['prediction_timeline'])
            fig = px.line(df, x='time', y='value',
                         title="Tahmin Değişim Grafiği",
                         labels={'time': 'Zaman', 'value': 'Tahmin Değeri'})
            st.plotly_chart(fig, use_container_width=True, key=f"timeline_{fixture_id}")

    # Öğrenme önerileri
    suggestions = st.session_state.simulation_engine.get_learning_suggestions(simulation_id)
    if suggestions:
        st.markdown("### 💡 Geliştirme Önerileri")
        for suggestion in suggestions:
            st.info(suggestion)


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