import streamlit as st
import time
from data_handler import DataHandler
from models import StatisticalModel
from utils import create_probability_chart, create_form_chart
from strategy_advisor import StrategyAdvisor
from match_commentator import MatchCommentator

st.set_page_config(page_title="Canlı Maçlar", layout="wide")

# Initialize session state
if 'data_handler' not in st.session_state:
    try:
        api_key = st.secrets.get("RAPIDAPI_KEY")
        if api_key:
            st.session_state.data_handler = DataHandler(api_key)
            st.session_state.statistical_model = StatisticalModel()
            st.session_state.strategy_advisor = StrategyAdvisor(None)  # Will be updated with data
        else:
            st.error("API anahtarı bulunamadı. Lütfen RAPIDAPI_KEY'i Secrets kısmına ekleyin.")
            st.stop()
    except Exception as e:
        st.error(f"API bağlantısı sırasında hata oluştu: {str(e)}")
        st.stop()

# Initialize commentator separately to handle any potential errors
if 'commentator' not in st.session_state:
    try:
        st.session_state.commentator = MatchCommentator()
    except Exception as e:
        st.error(f"Yorumlayıcı başlatılırken hata oluştu: {str(e)}")
        st.session_state.commentator = None

def format_event(event):
    """Format match event for display"""
    return f"{event['time']['elapsed']}' - {event['team']['name']}: {event['type']} - {event['player']['name']}"

def calculate_live_win_probability(stats, score):
    """Calculate live win probability based on current stats and score"""
    home_score, away_score = score

    # Convert stats to numerical values for analysis
    home_stats = {
        'possession': float(stats[0]['statistics'][9]['value'].strip('%')) / 100 if stats else 0.5,
        'shots_on_target': int(stats[0]['statistics'][2]['value'] or 0) if stats else 0,
        'dangerous_attacks': int(stats[0]['statistics'][13]['value'] or 0) if stats else 0
    }

    away_stats = {
        'possession': float(stats[1]['statistics'][9]['value'].strip('%')) / 100 if stats else 0.5,
        'shots_on_target': int(stats[1]['statistics'][2]['value'] or 0) if stats else 0,
        'dangerous_attacks': int(stats[1]['statistics'][13]['value'] or 0) if stats else 0
    }

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

def display_match_details(fixture_id, match_info):
    """Display detailed statistics and events for a match"""
    with st.spinner("Maç detayları yükleniyor..."):
        # Get match statistics
        stats = st.session_state.data_handler.get_match_statistics(fixture_id)
        events = st.session_state.data_handler.get_match_events(fixture_id)

        # Score information
        score = [match_info['goals']['home'], match_info['goals']['away']]

        # Calculate live win probabilities
        win_probs = calculate_live_win_probability(stats, score)

        # AI Commentary Section
        if st.session_state.commentator is not None:
            st.subheader("Maç Yorumu")
            commentary = st.session_state.commentator.generate_match_commentary(stats, score, events)
            st.markdown(f"💬 {commentary}")

            # Display prediction explanation
            prediction_explanation = st.session_state.commentator.explain_prediction(win_probs, stats)
            st.info(f"📊 **Tahmin Açıklaması:** {prediction_explanation}")

        # Display win probability chart
        st.subheader("Canlı Kazanma Olasılıkları")
        st.plotly_chart(create_probability_chart(
            match_info['teams']['home']['name'],
            match_info['teams']['away']['name'],
            win_probs,
            "Canlı Tahmin"
        ), use_container_width=True)

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

            # Performance Analysis
            st.subheader("Performans Analizi")

            # Calculate performance metrics
            home_possession = float(stats[0]['statistics'][9]['value'].strip('%')) if stats[0]['statistics'][9]['value'] else 50
            home_shots = int(stats[0]['statistics'][2]['value'] or 0)
            home_attacks = int(stats[0]['statistics'][13]['value'] or 0)

            away_possession = float(stats[1]['statistics'][9]['value'].strip('%')) if stats[1]['statistics'][9]['value'] else 50
            away_shots = int(stats[1]['statistics'][2]['value'] or 0)
            away_attacks = int(stats[1]['statistics'][13]['value'] or 0)

            # Display performance metrics
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Hücum Etkinliği**")
                st.progress(home_shots / max(home_shots + away_shots, 1), 
                          text=f"İsabetli Şut Etkinliği: {home_shots}")
                st.progress(home_attacks / max(home_attacks + away_attacks, 1),
                          text=f"Tehlikeli Atak: {home_attacks}")

            with col2:
                st.write("**Top Kontrolü**")
                st.progress(home_possession / 100,
                          text=f"Top Kontrolü: %{home_possession:.1f}")

        if events:
            st.subheader("Önemli Olaylar")
            for event in events:
                if event['type'] == 'Goal':
                    st.success(format_event(event))
                elif event['type'] in ['Card', 'subst']:
                    st.warning(format_event(event))
                else:
                    st.info(format_event(event))

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
    st.error(f"Bir hata oluştu: {str(e)}")
    st.info("Sayfayı yenilemek için F5 tuşuna basın veya sayfayı manuel olarak yenileyin.")