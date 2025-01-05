from data_handler import DataHandler
from ml_predictor import MLPredictor
import numpy as np
import json

def create_training_data():
    # Eğitim verisi oluştur
    X = []
    y = []
    
    # 100 örnek veri oluştur
    for _ in range(100):
        match_features = []
        for match in range(10):  # Her örnek için 10 maç
            # Temel istatistikler
            goals_scored = np.random.poisson(1.5)  # Ortalama 1.5 gol
            goals_conceded = np.random.poisson(1.2)  # Ortalama 1.2 gol
            possession = np.random.normal(50, 10)  # Ortalama 50, std 10
            shots = np.random.poisson(12)  # Ortalama 12 şut
            shots_on_target = min(shots, np.random.poisson(5))  # Ortalama 5 isabetli şut
            corners = np.random.poisson(5)  # Ortalama 5 korner
            fouls = np.random.poisson(12)  # Ortalama 12 faul
            yellow_cards = np.random.poisson(2)  # Ortalama 2 sarı kart
            red_cards = np.random.binomial(1, 0.05)  # %5 kırmızı kart olasılığı
            
            # Form ve momentum
            team_rating = np.random.normal(7, 1)  # 1-10 arası takım puanı
            first_half_goals = np.random.binomial(goals_scored, 0.6)  # %60 ilk yarı gol olasılığı
            second_half_goals = goals_scored - first_half_goals
            late_goals = np.random.binomial(second_half_goals, 0.3)  # %30 son dakika gol olasılığı
            
            # Taktik ve performans
            win_streak = np.random.poisson(1)  # Galibiyet serisi
            clean_sheets = np.random.binomial(1, 0.3)  # %30 gol yememe olasılığı
            formation_success = np.random.normal(0.7, 0.1)  # Formasyon başarısı
            counter_goals = np.random.binomial(goals_scored, 0.2)  # %20 kontra gol olasılığı
            set_piece_goals = np.random.binomial(goals_scored, 0.25)  # %25 duran top gol olasılığı
            possession_score = np.clip(np.random.normal(0.6, 0.1), 0, 1)  # Top kontrolü skoru
            defensive_rating = np.clip(np.random.normal(7.5, 1), 0, 10)  # Defans performansı
            
            match_stats = [
                goals_scored, goals_conceded, possession, shots_on_target,
                shots, corners, fouls, yellow_cards, red_cards, team_rating,
                first_half_goals, second_half_goals, late_goals, win_streak,
                clean_sheets, formation_success, counter_goals, set_piece_goals,
                possession_score, defensive_rating
            ]
            match_features.extend(match_stats)
        
        X.append(match_features)
        
        # Hedef değer (sonraki maçtaki gol sayısı)
        next_match_goals = np.random.poisson(
            max(0.5, min(4, goals_scored * 0.7 + team_rating * 0.3))
        )
        y.append(next_match_goals)
    
    return np.array(X), np.array(y)

def test_prediction():
    # DataHandler ve MLPredictor'ı başlat
    dh = DataHandler()
    mlp = MLPredictor()

    # Eğitim verisi oluştur ve modeli eğit
    X_train, y_train = create_training_data()
    mlp.xgb_model1.fit(X_train, y_train)
    mlp.xgb_model2.fit(X_train, y_train)

    # Örnek maç verisi
    match_stats = {
        0: {
            'team': {'id': 1, 'name': 'Fenerbahçe'},
            'statistics': [
                {'type': 'Shots on Goal', 'value': str(np.random.poisson(6))},  # Daha gerçekçi değerler
                {'type': 'Shots off Goal', 'value': str(np.random.poisson(8))},
                {'type': 'Total Shots', 'value': str(np.random.poisson(15))},
                {'type': 'Blocked Shots', 'value': str(np.random.poisson(3))},
                {'type': 'Corner Kicks', 'value': str(np.random.poisson(6))},
                {'type': 'Fouls', 'value': str(np.random.poisson(12))},
                {'type': 'Ball Possession', 'value': f"{np.random.normal(55, 5):.0f}%"},
                {'type': 'Yellow Cards', 'value': str(np.random.poisson(2))},
                {'type': 'Red Cards', 'value': str(np.random.binomial(1, 0.05))},
                {'type': 'Total passes', 'value': str(np.random.poisson(450))},
                {'type': 'Passes accurate', 'value': str(np.random.poisson(380))},
                {'type': 'Passes %', 'value': f"{np.random.normal(84, 3):.0f}%"}
            ]
        },
        1: {
            'team': {'id': 2, 'name': 'Galatasaray'},
            'statistics': [
                {'type': 'Shots on Goal', 'value': str(np.random.poisson(5))},
                {'type': 'Shots off Goal', 'value': str(np.random.poisson(7))},
                {'type': 'Total Shots', 'value': str(np.random.poisson(13))},
                {'type': 'Blocked Shots', 'value': str(np.random.poisson(2))},
                {'type': 'Corner Kicks', 'value': str(np.random.poisson(5))},
                {'type': 'Fouls', 'value': str(np.random.poisson(11))},
                {'type': 'Ball Possession', 'value': f"{np.random.normal(45, 5):.0f}%"},
                {'type': 'Yellow Cards', 'value': str(np.random.poisson(2))},
                {'type': 'Red Cards', 'value': str(np.random.binomial(1, 0.05))},
                {'type': 'Total passes', 'value': str(np.random.poisson(380))},
                {'type': 'Passes accurate', 'value': str(np.random.poisson(310))},
                {'type': 'Passes %', 'value': f"{np.random.normal(82, 3):.0f}%"}
            ]
        }
    }

    # Örnek olaylar - daha gerçekçi zaman dağılımı
    events = []
    for _ in range(np.random.poisson(3)):  # Ortalama 3 önemli olay
        event_time = np.random.randint(1, 90)
        event_team = np.random.choice([1, 2])
        event_type = np.random.choice(['Goal', 'Shot', 'Corner', 'Card'], p=[0.2, 0.4, 0.25, 0.15])
        events.append({
            'time': {'elapsed': event_time},
            'team': {'id': event_team, 'name': 'Fenerbahçe' if event_team == 1 else 'Galatasaray'},
            'type': event_type
        })
    events.sort(key=lambda x: x['time']['elapsed'])  # Olayları zamana göre sırala

    # Geçmiş veriler - daha gerçekçi değerler
    historical_data = {
        'recent_matches': []
    }
    for _ in range(5):  # Son 5 maç
        match_data = {
            'goals_scored': np.random.poisson(1.5),
            'goals_conceded': np.random.poisson(1.2),
            'possession': np.random.normal(52, 8),
            'shots_on_target': np.random.poisson(5),
            'shots_total': np.random.poisson(13),
            'corners': np.random.poisson(5),
            'fouls': np.random.poisson(12),
            'yellow_cards': np.random.poisson(2),
            'red_cards': np.random.binomial(1, 0.05),
            'team_rating': np.clip(np.random.normal(7.5, 0.8), 0, 10),
            'first_half_goals': np.random.poisson(0.8),
            'second_half_goals': np.random.poisson(0.7),
            'late_goals': np.random.poisson(0.3),
            'win_streak': np.random.poisson(1),
            'clean_sheets': np.random.binomial(1, 0.3),
            'formation_success_rate': np.clip(np.random.normal(0.7, 0.1), 0, 1),
            'counter_attack_goals': np.random.poisson(0.5),
            'set_piece_goals': np.random.poisson(0.4),
            'possession_score': np.clip(np.random.normal(0.6, 0.1), 0, 1),
            'defensive_rating': np.clip(np.random.normal(7.2, 0.9), 0, 10)
        }
        historical_data['recent_matches'].append(match_data)

    # Tahmin yap
    prediction = mlp.predict_goals(match_stats, events, historical_data)
    
    print("\nTahmin Sonuçları:")
    print("-" * 50)
    print(f"Tahmini Gol Sayısı: {prediction['predicted_goals']:.2f}")
    print(f"Model 1 Güven Skoru: {prediction['model1_confidence']:.2%}")
    print(f"Model 2 Güven Skoru: {prediction['model2_confidence']:.2%}")
    print(f"Ensemble Güven Skoru: {prediction['ensemble_confidence']:.2%}")
    
    # Maç detaylarını göster
    print("\nMaç Detayları:")
    print("-" * 50)
    print("Fenerbahçe:")
    print(f"Şutlar: {match_stats[0]['statistics'][2]['value']}")
    print(f"Kornerler: {match_stats[0]['statistics'][4]['value']}")
    print(f"Sarı Kartlar: {match_stats[0]['statistics'][7]['value']}")
    print("\nGalatasaray:")
    print(f"Şutlar: {match_stats[1]['statistics'][2]['value']}")
    print(f"Kornerler: {match_stats[1]['statistics'][4]['value']}")
    print(f"Sarı Kartlar: {match_stats[1]['statistics'][7]['value']}")

if __name__ == "__main__":
    test_prediction() 