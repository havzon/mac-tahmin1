import streamlit as st
import time
from data_handler import DataHandler

st.set_page_config(page_title="Canlı Maçlar", layout="wide")

# Initialize session state
if 'data_handler' not in st.session_state:
    try:
        api_key = st.secrets.get("RAPIDAPI_KEY")
        if api_key:
            st.session_state.data_handler = DataHandler(api_key)
        else:
            st.error("API anahtarı bulunamadı. Lütfen RAPIDAPI_KEY'i Secrets kısmına ekleyin.")
            st.stop()
    except Exception as e:
        st.error(f"API bağlantısı sırasında hata oluştu: {str(e)}")
        st.stop()

def format_event(event):
    """Format match event for display"""
    return f"{event['time']['elapsed']}' - {event['team']['name']}: {event['type']} - {event['player']['name']}"

def display_match_details(fixture_id):
    """Display detailed statistics and events for a match"""
    with st.spinner("Maç detayları yükleniyor..."):
        # Get match statistics
        stats = st.session_state.data_handler.get_match_statistics(fixture_id)
        if stats:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Ev Sahibi İstatistikleri")
                for stat in stats[0]['statistics']:
                    st.write(f"{stat['type']}: {stat['value']}")

            with col2:
                st.subheader("Deplasman İstatistikleri")
                for stat in stats[1]['statistics']:
                    st.write(f"{stat['type']}: {stat['value']}")

        # Get match events
        events = st.session_state.data_handler.get_match_events(fixture_id)
        if events:
            st.subheader("Maç Olayları")
            for event in events:
                st.info(format_event(event))

        # Get match lineups
        lineups = st.session_state.data_handler.get_match_lineups(fixture_id)
        if lineups:
            st.subheader("Kadrolar")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Ev Sahibi Kadro**")
                st.write(f"Diziliş: {lineups[0]['formation']}")
                st.write("İlk 11:")
                for player in lineups[0]['startXI']:
                    st.write(f"• {player['player']['name']}")

            with col2:
                st.write("**Deplasman Kadro**")
                st.write(f"Diziliş: {lineups[1]['formation']}")
                st.write("İlk 11:")
                for player in lineups[1]['startXI']:
                    st.write(f"• {player['player']['name']}")

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
                display_match_details(match['fixture']['id'])
    else:
        st.info("Şu anda canlı maç bulunmuyor.")

    # Auto refresh logic
    if auto_refresh:
        time.sleep(30)
        st.experimental_rerun()

except Exception as e:
    st.error(f"Bir hata oluştu: {str(e)}")
    st.info("Sayfayı yenilemek için F5 tuşuna basın veya sayfayı manuel olarak yenileyin.")