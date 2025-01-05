from typing import List, Dict
import requests
import numpy as np
import streamlit as st
import http.client

def get_live_matches() -> List[Dict]:
    """Canlı maçları getir ve istatistik durumunu kontrol et"""
    try:
        # API'den canlı maçları çek
        response = requests.get(
            "https://api-football-v1.p.rapidapi.com/v3/fixtures?live=all",
            headers={
                "X-RapidAPI-Key": "c417cb5967msh54c12f850f3798cp12a733jsn315fab53ee3c",
                "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
            }
        )
        response.raise_for_status()
        
        matches = response.json()
        
        # Her maç için istatistik durumunu kontrol et
        for match in matches:
            match['statistics_available'] = bool(match.get('statistics'))
            
            # Eğer istatistik yoksa örnek veri oluştur
            if not match['statistics_available']:
                match['statistics'] = generate_sample_stats()
                
        return matches
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Canlı maçlar alınırken hata: {str(e)}")
        return []

def generate_sample_stats() -> Dict:
    """Örnek maç istatistikleri oluştur"""
    return {
        'home_team': {
            'shots_on_target': np.random.randint(0, 10),
            'shots_off_target': np.random.randint(0, 15),
            'possession': f"{np.random.randint(40, 60)}%",
            'corners': np.random.randint(0, 10),
            'fouls': np.random.randint(5, 20)
        },
        'away_team': {
            'shots_on_target': np.random.randint(0, 10),
            'shots_off_target': np.random.randint(0, 15),
            'possession': f"{np.random.randint(40, 60)}%",
            'corners': np.random.randint(0, 10),
            'fouls': np.random.randint(5, 20)
        }
    }

def display_live_matches(matches: List[Dict]):
    """Canlı maçları göster"""
    if not matches:
        st.warning("Şu anda canlı maç bulunmamaktadır.")
        return
        
    for match in matches:
        # Maç bilgilerini göster
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(f"**{match['home_team']['name']}** vs **{match['away_team']['name']}**")
            st.write(f"Skor: {match['score']} ({match['minute']}')")
            
        with col2:
            if not match['statistics_available']:
                st.info("Maç istatistikleri şu an için mevcut değil. Tahminler sınırlı veriyle oluşturuluyor.")
            else:
                display_match_stats(match['statistics']) 

def get_odds_data():
    try:
        conn = http.client.HTTPSConnection("api-football-v1.p.rapidapi.com")
        
        headers = {
            'x-rapidapi-key': "c417cb5967msh54c12f850f3798cp12a733jsn315fab53ee3c",
            'x-rapidapi-host': "api-football-v1.p.rapidapi.com"
        }
        
        conn.request("GET", "/v2/odds/league/865927/bookmaker/5?page=2", headers=headers)
        
        res = conn.getresponse()
        
        if res.status == 200:
            data = res.read()
            return data.decode("utf-8")
        else:
            print(f"Hata: {res.status} - {res.reason}")
            return None
            
    except Exception as e:
        print(f"API bağlantısı sırasında hata oluştu: {str(e)}")
        return None

# Kullanım örneği
odds_data = get_odds_data()
if odds_data:
    print(odds_data) 