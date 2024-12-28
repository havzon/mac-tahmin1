import os
from typing import Dict, List, Optional
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchCommentator:
    def __init__(self):
        """Maç yorumlayıcı ve tahmin açıklayıcı"""
        logger.info("Initializing MatchCommentator")
        try:
            self.initialized = True
            logger.info("MatchCommentator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MatchCommentator: {str(e)}")
            self.initialized = False
            raise

    def generate_match_commentary(self, match_stats: Dict, score: List[int], events: List[Dict]) -> str:
        """Maç istatistiklerine ve olaylara göre yorum üret"""
        try:
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
            if match_stats and len(match_stats) >= 2:
                logger.info("Processing match statistics")
                try:
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
                except Exception as e:
                    logger.error(f"Error processing match statistics: {str(e)}")
                    commentary.append("İstatistik analizi yapılırken bir hata oluştu.")

            # Son olayların analizi
            if events:
                logger.info("Processing recent events")
                try:
                    recent_events = events[-3:]  # Son 3 olay
                    for event in recent_events:
                        if event['type'] == 'Goal':
                            commentary.append(f"⚽ {event['time']['elapsed']}. dakikada {event['team']['name']} golü buldu!")
                        elif event['type'] == 'Card':
                            commentary.append(f"🟨 {event['time']['elapsed']}. dakikada kart görüldü, oyun sertleşiyor.")
                except Exception as e:
                    logger.error(f"Error processing events: {str(e)}")
                    commentary.append("Maç olayları analiz edilirken bir hata oluştu.")

            return " ".join(commentary)
        except Exception as e:
            logger.error(f"Error generating match commentary: {str(e)}")
            return "Maç yorumu oluşturulurken bir hata meydana geldi."

    def predict_next_goal(self, match_stats: Dict, events: List[Dict]) -> Dict:
        """Bir sonraki golü kimin atacağını tahmin et"""
        try:
            if not match_stats or len(match_stats) < 2:
                logger.warning("Insufficient match statistics for prediction")
                return {
                    'prediction': 'Tahmin için yeterli veri yok',
                    'probability': 0.0,
                    'expected_time': None,
                    'confidence': 'düşük'
                }

            # İstatistikleri sayısal değerlere dönüştür
            try:
                home_stats = self._extract_team_stats(match_stats[0]['statistics'])
                away_stats = self._extract_team_stats(match_stats[1]['statistics'])
            except (IndexError, KeyError, ValueError) as e:
                logger.error(f"Error parsing statistics: {str(e)}")
                return {
                    'prediction': 'İstatistik verilerinde hata',
                    'probability': 0.0,
                    'expected_time': None,
                    'confidence': 'düşük'
                }

            # Gelişmiş momentum hesaplama
            home_momentum = self._calculate_team_momentum(home_stats, events, match_stats[0]['team']['name'], True)
            away_momentum = self._calculate_team_momentum(away_stats, events, match_stats[1]['team']['name'], False)

            # Momentum normalizasyonu
            total_momentum = home_momentum + away_momentum
            if total_momentum == 0:
                home_prob = away_prob = 0.5
            else:
                home_prob = home_momentum / total_momentum
                away_prob = away_momentum / total_momentum

            # Tahmini gol zamanı
            expected_time = self._predict_next_goal_time(events)

            # Güven seviyesi hesaplama
            confidence = self._calculate_prediction_confidence(max(home_prob, away_prob), match_stats)

            # Tahmin sonucu
            threshold = 0.55
            if home_prob > away_prob and home_prob > threshold:
                return {
                    'prediction': 'Ev sahibi takım gol atabilir',
                    'probability': home_prob,
                    'expected_time': expected_time,
                    'confidence': confidence,
                    'details': self._get_prediction_details(home_stats, 'ev sahibi')
                }
            elif away_prob > home_prob and away_prob > threshold:
                return {
                    'prediction': 'Deplasman takımı gol atabilir',
                    'probability': away_prob,
                    'expected_time': expected_time,
                    'confidence': confidence,
                    'details': self._get_prediction_details(away_stats, 'deplasman')
                }
            else:
                return {
                    'prediction': 'Şu an için gol beklentisi düşük',
                    'probability': max(home_prob, away_prob),
                    'expected_time': None,
                    'confidence': 'düşük'
                }

        except Exception as e:
            logger.error(f"Error predicting next goal: {str(e)}")
            return {
                'prediction': 'Tahmin hesaplanırken hata oluştu',
                'probability': 0.0,
                'expected_time': None,
                'confidence': 'düşük'
            }

    def _extract_team_stats(self, stats: List[Dict]) -> Dict:
        """Takım istatistiklerini çıkar ve sayısallaştır"""
        try:
            return {
                'shots': int(stats[2]['value'] or 0),
                'possession': float(stats[9]['value'].strip('%')) if stats[9]['value'] else 50,
                'attacks': int(stats[13]['value'] or 0),
                'corners': int(stats[6]['value'] or 0),
                'on_target': int(stats[0]['value'] or 0),
                'blocked_shots': int(stats[1]['value'] or 0),
                'fouls': int(stats[7]['value'] or 0)
            }
        except (IndexError, ValueError) as e:
            logger.error(f"Error extracting team stats: {str(e)}")
            return {
                'shots': 0, 'possession': 50, 'attacks': 0,
                'corners': 0, 'on_target': 0, 'blocked_shots': 0, 'fouls': 0
            }

    def _calculate_team_momentum(self, stats: Dict, events: List[Dict], team_name: str, is_home: bool) -> float:
        """Gelişmiş takım momentumu hesaplama"""
        try:
            # Temel istatistik ağırlıkları
            momentum = (
                stats['shots'] * 0.15 +  # İsabetli şutlar
                (stats['possession'] / 100) * 0.15 +  # Top kontrolü
                (stats['attacks'] / 100) * 0.2 +  # Tehlikeli ataklar
                (stats['corners'] * 0.1) +  # Kornerler
                (stats['on_target'] * 0.2) +  # İsabetli şutlar
                (1 - stats['fouls'] / 20) * 0.1  # Faul etkisi (az faul = pozitif)
            )

            # Son olayların etkisi
            recent_momentum = self._calculate_recent_momentum(events, team_name)
            momentum += recent_momentum * 0.2

            # Ev sahibi avantajı
            if is_home:
                momentum *= 1.05

            return max(0.0, min(1.0, momentum))

        except Exception as e:
            logger.error(f"Error calculating team momentum: {str(e)}")
            return 0.5

    def _calculate_recent_momentum(self, events: List[Dict], team_name: str) -> float:
        """Son dakikalardaki momentum hesaplama"""
        try:
            if not events:
                return 0.0

            momentum = 0.0
            recent_events = events[-5:]  # Son 5 olay

            for event in recent_events:
                if event['team']['name'] == team_name:
                    if event['type'] == 'Goal':
                        momentum += 0.3
                    elif event['type'] == 'Card':
                        momentum -= 0.1
                    elif event['type'] == 'subst':
                        momentum += 0.05
                    elif event['type'] == 'Var':
                        momentum += 0.05

            return min(1.0, max(0.0, momentum))

        except Exception as e:
            logger.error(f"Error calculating recent momentum: {str(e)}")
            return 0.0

    def _predict_next_goal_time(self, events: List[Dict]) -> Optional[int]:
        """Sonraki golün tahmini zamanını hesapla"""
        try:
            if not events:
                return None

            # Gol aralıklarını hesapla
            goal_intervals = []
            prev_goal_time = 0
            last_goal_time = 0

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
                predicted_time = last_goal_time + max(10, min(30, avg_interval))
                return min(90, int(predicted_time))
            else:
                current_time = events[-1]['time']['elapsed']
                return min(90, current_time + np.random.randint(15, 25))

        except Exception as e:
            logger.error(f"Error predicting next goal time: {str(e)}")
            return None

    def _calculate_prediction_confidence(self, probability: float, match_stats: Dict) -> str:
        """Tahmin güven seviyesini hesapla"""
        try:
            if probability > 0.7:
                return 'yüksek'
            elif probability > 0.6:
                return 'orta'
            else:
                return 'düşük'
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {str(e)}")
            return 'düşük'

    def _get_prediction_details(self, stats: Dict, team_type: str) -> str:
        """Tahmin detaylarını açıkla"""
        try:
            details = []

            if stats['shots'] > 3:
                details.append(f"{team_type} takım {stats['shots']} şutla etkili")

            if stats['possession'] > 60:
                details.append("top kontrolünü elinde tutuyor")

            if stats['attacks'] > 20:
                details.append("sürekli atak yapıyor")

            if not details:
                return "Standart oyun akışı devam ediyor"

            return f"{', '.join(details)}."
        except Exception as e:
            logger.error(f"Error getting prediction details: {str(e)}")
            return "Detaylı analiz yapılamadı"

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