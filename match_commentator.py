import os
from typing import Dict, List, Optional
import numpy as np

class MatchCommentator:
    def __init__(self):
        """Maç yorumlayıcı ve tahmin açıklayıcı"""
        pass

    def generate_match_commentary(self, match_stats: Dict, score: List[int], events: List[Dict]) -> str:
        """Maç istatistiklerine ve olaylara göre yorum üret"""
        commentary = []

        # Skor analizi
        home_score, away_score = score
        if home_score > away_score:
            score_diff = home_score - away_score
            if score_diff >= 3:
                commentary.append("Ev sahibi takım maça tam hakimiyet kurmuş durumda.")
            else:
                commentary.append("Ev sahibi takım önde, ancak maç hala dengeli.")
        elif away_score > home_score:
            score_diff = away_score - home_score
            if score_diff >= 3:
                commentary.append("Deplasman takımı sahada üstünlüğü ele geçirmiş görünüyor.")
            else:
                commentary.append("Deplasman takımı önde, fakat maç henüz bitmedi.")
        else:
            commentary.append("Şu an için skorlar eşit, her iki takım da üstünlük kurmaya çalışıyor.")

        # İstatistik analizi
        if match_stats:
            home_stats = match_stats[0]['statistics']
            away_stats = match_stats[1]['statistics']

            # Top kontrolü analizi
            home_possession = float(home_stats[9]['value'].strip('%')) if home_stats[9]['value'] else 50
            if abs(home_possession - 50) > 10:
                if home_possession > 50:
                    commentary.append(f"Ev sahibi takım %{home_possession:.0f} top kontrolüyle oyunu yönlendiriyor.")
                else:
                    commentary.append(f"Deplasman takımı %{100-home_possession:.0f} top kontrolüyle oyuna hakim.")

            # Şut analizi
            home_shots = int(home_stats[2]['value'] or 0)
            away_shots = int(away_stats[2]['value'] or 0)
            if abs(home_shots - away_shots) > 3:
                if home_shots > away_shots:
                    commentary.append(f"Ev sahibi {home_shots} isabetli şutla rakibinden daha etkili.")
                else:
                    commentary.append(f"Deplasman {away_shots} isabetli şutla pozisyonları değerlendirmede daha başarılı.")

        # Son olayların analizi
        if events:
            recent_events = events[-3:]  # Son 3 olay
            for event in events[-3:]:
                if event['type'] == 'Goal':
                    commentary.append(f"⚽ {event['time']['elapsed']}. dakikada {event['team']['name']} golü buldu!")
                elif event['type'] == 'Card':
                    commentary.append(f"🟨 {event['time']['elapsed']}. dakikada kart görüldü, oyun sertleşiyor.")

        return " ".join(commentary)

    def predict_next_goal(self, match_stats: Dict, events: List[Dict]) -> Dict:
        """Bir sonraki golü kimin atacağını tahmin et"""
        if not match_stats:
            return {
                'prediction': 'Tahmin için yeterli veri yok',
                'probability': 0.0,
                'expected_time': None
            }

        home_stats = match_stats[0]['statistics']
        away_stats = match_stats[1]['statistics']

        # İstatistikleri sayısal değerlere dönüştür
        home_shots = int(home_stats[2]['value'] or 0)
        away_shots = int(away_stats[2]['value'] or 0)
        home_possession = float(home_stats[9]['value'].strip('%')) if home_stats[9]['value'] else 50
        home_attacks = int(home_stats[13]['value'] or 0)
        away_attacks = int(away_stats[13]['value'] or 0)
        home_corners = int(home_stats[6]['value'] or 0)
        away_corners = int(away_stats[6]['value'] or 0)

        # Gelişmiş momentum hesaplama
        home_momentum = (
            home_shots * 0.25 +  # İsabetli şutlar
            (home_possession / 100) * 0.2 +  # Top kontrolü
            (home_attacks / max(home_attacks + away_attacks, 1)) * 0.3 +  # Tehlikeli ataklar
            (home_corners / max(home_corners + away_corners, 1)) * 0.15 +  # Korner etkinliği
            self._calculate_recent_momentum(events, 'home') * 0.1  # Son dakika momentumu
        )

        away_momentum = (
            away_shots * 0.25 +
            ((100 - home_possession) / 100) * 0.2 +
            (away_attacks / max(home_attacks + away_attacks, 1)) * 0.3 +
            (away_corners / max(home_corners + away_corners, 1)) * 0.15 +
            self._calculate_recent_momentum(events, 'away') * 0.1
        )

        # Skor etkisi
        if events:
            last_goal = next((event for event in reversed(events) if event['type'] == 'Goal'), None)
            if last_goal:
                team_scored_last = 'home' if last_goal['team']['name'] == match_stats[0]['team']['name'] else 'away'
                if team_scored_last == 'home':
                    home_momentum *= 1.1  # Son golü atan takıma bonus
                    away_momentum *= 1.15  # Yenik takıma hafif bonus
                else:
                    away_momentum *= 1.1
                    home_momentum *= 1.15

        # Olasılık normalizasyonu
        total_momentum = home_momentum + away_momentum
        if total_momentum == 0:
            home_prob = away_prob = 0.5
        else:
            home_prob = home_momentum / total_momentum
            away_prob = away_momentum / total_momentum

        # Tahmini gol zamanı hesaplama
        expected_time = self._predict_next_goal_time(events)

        # Tahmin sonucu
        if home_prob > away_prob and home_prob > 0.55:
            return {
                'prediction': 'Ev sahibi takım gol atabilir',
                'probability': home_prob,
                'expected_time': expected_time,
                'confidence': self._calculate_prediction_confidence(home_prob, match_stats)
            }
        elif away_prob > home_prob and away_prob > 0.55:
            return {
                'prediction': 'Deplasman takımı gol atabilir',
                'probability': away_prob,
                'expected_time': expected_time,
                'confidence': self._calculate_prediction_confidence(away_prob, match_stats)
            }
        else:
            return {
                'prediction': 'Şu an için gol beklentisi düşük',
                'probability': max(home_prob, away_prob),
                'expected_time': None,
                'confidence': 'düşük'
            }

    def _calculate_recent_momentum(self, events: List[Dict], team: str) -> float:
        """Son dakikalardaki momentum hesaplama"""
        if not events:
            return 0.0

        momentum = 0.0
        recent_events = events[-5:]  # Son 5 olay
        for event in recent_events:
            event_team = 'home' if event['team']['name'] == events[0]['team']['home'] else 'away'
            if event_team == team:
                if event['type'] == 'Goal':
                    momentum += 0.3
                elif event['type'] == 'Card':
                    momentum -= 0.1
                elif event['type'] in ['subst', 'Var']:
                    momentum += 0.05

        return min(1.0, max(0.0, momentum))

    def _predict_next_goal_time(self, events: List[Dict]) -> Optional[int]:
        """Sonraki golün tahmini zamanını hesapla"""
        if not events:
            return None

        # Son golün zamanını bul
        last_goal_time = 0
        goal_intervals = []
        prev_goal_time = 0

        for event in events:
            if event['type'] == 'Goal':
                current_time = event['time']['elapsed']
                if prev_goal_time > 0:
                    goal_intervals.append(current_time - prev_goal_time)
                prev_goal_time = current_time
                last_goal_time = current_time

        # Ortalama gol aralığını hesapla
        if goal_intervals:
            avg_interval = sum(goal_intervals) / len(goal_intervals)
            # Tahmini zamanı hesapla ve normalize et
            predicted_time = last_goal_time + max(10, min(30, avg_interval))
            return min(90, int(predicted_time))
        else:
            # Hiç gol yoksa, maçın durumuna göre tahmin yap
            current_time = events[-1]['time']['elapsed']
            return min(90, current_time + np.random.randint(15, 25))

    def _calculate_prediction_confidence(self, probability: float, match_stats: Dict) -> str:
        """Tahmin güven seviyesini hesapla"""
        if probability > 0.7:
            return 'yüksek'
        elif probability > 0.6:
            return 'orta'
        else:
            return 'düşük'

    def explain_prediction(self, win_probs: List[float], match_stats: Dict) -> str:
        """Tahmin olasılıklarını açıkla"""
        home_prob, draw_prob, away_prob = win_probs
        explanation = []

        # En yüksek olasılığı bul
        max_prob = max(win_probs)
        if max_prob == home_prob:
            if home_prob > 0.5:
                explanation.append("Ev sahibi takım maçın favorisi olarak görünüyor.")
            else:
                explanation.append("Ev sahibi takım hafif favori.")
        elif max_prob == away_prob:
            if away_prob > 0.5:
                explanation.append("Deplasman takımı maçta öne çıkıyor.")
            else:
                explanation.append("Deplasman takımı az farkla favori.")
        else:
            explanation.append("Beraberlik ihtimali yüksek, dengeli bir maç bekleniyor.")

        # İstatistik bazlı açıklama
        if match_stats:
            home_stats = match_stats[0]['statistics']
            away_stats = match_stats[1]['statistics']

            # Tehlikeli atakları karşılaştır
            home_attacks = int(home_stats[13]['value'] or 0)
            away_attacks = int(away_stats[13]['value'] or 0)
            if abs(home_attacks - away_attacks) > 5:
                if home_attacks > away_attacks:
                    explanation.append(f"Ev sahibi takım {home_attacks} tehlikeli atakla baskı kuruyor.")
                else:
                    explanation.append(f"Deplasman takımı {away_attacks} tehlikeli atakla üstünlük sağlıyor.")

            # Şut isabetini karşılaştır
            home_shots = int(home_stats[2]['value'] or 0)
            away_shots = int(away_stats[2]['value'] or 0)
            if abs(home_shots - away_shots) > 2:
                if home_shots > away_shots:
                    explanation.append("Ev sahibi isabetli şutlarda daha etkili.")
                else:
                    explanation.append("Deplasman isabetli şutlarda öne çıkıyor.")

        # Genel değerlendirme
        if max_prob > 0.6:
            explanation.append("İstatistikler ve oyun gidişatı bu tahmini güçlü şekilde destekliyor.")
        elif max_prob > 0.4:
            explanation.append("Tahmin güvenilir görünüyor ancak sürprizlere açık bir maç.")
        else:
            explanation.append("Maçın gidişatı çok değişken, kesin bir tahmin yapmak zor.")

        return " ".join(explanation)