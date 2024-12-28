import os
from typing import Dict, List, Optional
import numpy as np
import logging
from ml_predictor import MLPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchCommentator:
    def __init__(self):
        """Maç yorumlayıcı ve tahmin açıklayıcı"""
        logger.info("Initializing MatchCommentator")
        try:
            self.initialized = True
            self.ml_predictor = MLPredictor()  # ML tahmin sistemini başlat
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

    def predict_next_goal(self, match_stats: Dict, events: List[Dict], betting_odds: Optional[Dict] = None) -> Dict:
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

            # ML tabanlı tahmin
            predictions = self.ml_predictor.predict_goals(match_stats, events)

            if not predictions or 'predictions' not in predictions:
                return {
                    'prediction': 'Tahmin hesaplanamadı',
                    'probability': 0.0,
                    'expected_time': None,
                    'confidence': 'düşük'
                }

            # Bahis oranları varsa tahminleri güncelle
            if betting_odds:
                try:
                    # Bahis oranlarından olasılıkları hesapla
                    total_prob = sum(1/float(odd) for odd in betting_odds.values())
                    implied_probs = {k: (1/float(v))/total_prob for k, v in betting_odds.items()}

                    # Her güven seviyesi için bahis oranlarını entegre et
                    for confidence_level in predictions['predictions']:
                        pred = predictions['predictions'][confidence_level]

                        # Bahis oranı bazlı güncelleme
                        if 'probability' in pred:
                            # Bahis oranı ağırlığını güven seviyesine göre ayarla
                            if confidence_level == 'yüksek':
                                betting_weight = 0.4
                            elif confidence_level == 'orta':
                                betting_weight = 0.3
                            else:
                                betting_weight = 0.2

                            ml_weight = 1 - betting_weight
                            pred['probability'] = (
                                pred['probability'] * ml_weight + 
                                implied_probs.get('next_goal', 0.5) * betting_weight
                            )

                        # Kalite faktörlerini güncelle
                        if 'quality_factors' not in pred:
                            pred['quality_factors'] = {}

                        pred['quality_factors'].update({
                            'betting_confidence': min(1.0, max(0.3, 1/total_prob)),
                            'odds_consistency': min(1.0, max(0.2, 1 - abs(pred['probability'] - implied_probs.get('next_goal', 0.5)))),
                            'market_strength': min(1.0, len(betting_odds) / 10)  # Bahis pazarı güçlülüğü
                        })

                        # Tahmin nedenini güncelle
                        reasons = []
                        if pred['probability'] > 0.6:
                            reasons.append("Yüksek olasılıklı gol beklentisi")
                        if pred['quality_factors']['betting_confidence'] > 0.7:
                            reasons.append("Güçlü bahis market göstergeleri")
                        if pred['quality_factors']['odds_consistency'] > 0.8:
                            reasons.append("Tutarlı tahmin ve oran verileri")

                        pred['reason'] = " & ".join(reasons) if reasons else "Standart analiz"

                except Exception as e:
                    logger.error(f"Error processing betting odds: {str(e)}")

            # En uygun tahmini seç
            recommended_level = predictions.get('recommended', 'düşük')
            selected_prediction = predictions['predictions'].get(recommended_level, {})

            if not selected_prediction:
                return {
                    'prediction': 'Tahmin seçilemedi',
                    'probability': 0.0,
                    'expected_time': None,
                    'confidence': 'düşük'
                }

            # Tahmin detaylarını hazırla
            probability = selected_prediction.get('probability', 0.0)
            confidence = selected_prediction.get('confidence', 'düşük')
            reason = selected_prediction.get('reason', '')

            # Tahmin açıklamasını oluştur
            if probability > 0.7:
                prediction = 'Yüksek gol olasılığı'
            elif probability > 0.4:
                prediction = 'Orta seviye gol olasılığı'
            else:
                prediction = 'Düşük gol olasılığı'

            # Maç durumu ve momentum bilgilerini ekle
            match_state = predictions.get('match_state', {})
            momentum = predictions.get('momentum', {})

            if match_state.get('phase') == 'son_dakikalar':
                prediction += ' (Maç sonu yaklaşıyor)'
            elif momentum.get('trend') in ['güçlü_ev_sahibi', 'güçlü_deplasman']:
                prediction += f" ({momentum['trend'].replace('_', ' ').title()})"

            # Bahis oranları varsa ekle
            prediction_data = {
                'prediction': prediction,
                'probability': probability,
                'expected_time': selected_prediction.get('expected_time'),
                'confidence': confidence,
                'reason': reason,
                'match_state': match_state,
                'momentum': momentum,
                'predictions': predictions['predictions']
            }

            if betting_odds:
                prediction_data.update({
                    'betting_odds': betting_odds,
                    'implied_probabilities': implied_probs
                })

            return prediction_data

        except Exception as e:
            logger.error(f"Error predicting next goal: {str(e)}")
            return {
                'prediction': 'Tahmin hesaplanırken hata oluştu',
                'probability': 0.0,
                'expected_time': None,
                'confidence': 'düşük',
                'error': str(e)
            }

    def _calculate_rule_based_prediction(self, home_stats: Dict, away_stats: Dict, 
                                      events: List[Dict], match_stats: Dict) -> Dict:
        """Kural tabanlı tahmin hesapla"""
        try:
            # Momentum hesaplama
            home_momentum = self._calculate_team_momentum(home_stats, events, match_stats[0]['team']['name'], True)
            away_momentum = self._calculate_team_momentum(away_stats, events, match_stats[1]['team']['name'], False)

            # Olasılık normalizasyonu
            total_momentum = home_momentum + away_momentum
            if total_momentum == 0:
                home_prob = away_prob = 0.5
            else:
                home_prob = home_momentum / total_momentum
                away_prob = away_momentum / total_momentum

            return {
                'home_prob': home_prob,
                'away_prob': away_prob,
                'expected_time': self._predict_next_goal_time(events)
            }

        except Exception as e:
            logger.error(f"Error in rule-based prediction: {str(e)}")
            return {'home_prob': 0.5, 'away_prob': 0.5, 'expected_time': None}

    def _combine_predictions(self, rule_based: Dict, ml_pred: Dict, 
                           home_stats: Dict, away_stats: Dict) -> Dict:
        """Kural tabanlı ve ML tahminlerini birleştir"""
        try:
            # ML tahmin ağırlığı (veri kalitesine göre)
            ml_weight = 0.6 if ml_pred['confidence'] != 'düşük' else 0.3
            rule_weight = 1 - ml_weight

            # Birleşik olasılık hesaplama
            combined_home_prob = (rule_based['home_prob'] * rule_weight + 
                                ml_pred['goal_probability'] * ml_weight)
            combined_away_prob = (rule_based['away_prob'] * rule_weight + 
                                ml_pred['goal_probability'] * ml_weight)

            # En olası sonucu belirle
            max_prob = max(combined_home_prob, combined_away_prob)
            threshold = 0.55

            if max_prob > threshold:
                if combined_home_prob > combined_away_prob:
                    prediction = 'Ev sahibi takım gol atabilir'
                    final_prob = combined_home_prob
                    details = self._get_prediction_details(home_stats, 'ev sahibi')
                else:
                    prediction = 'Deplasman takımı gol atabilir'
                    final_prob = combined_away_prob
                    details = self._get_prediction_details(away_stats, 'deplasman')
            else:
                prediction = 'Şu an için gol beklentisi düşük'
                final_prob = max_prob
                details = "Standart oyun akışı devam ediyor"

            # Expected time - ML ve kural tabanlı tahminlerin ortalaması
            expected_time = rule_based['expected_time']
            if ml_pred['expected_time']:
                expected_time = int((expected_time + ml_pred['expected_time']) / 2) if expected_time else ml_pred['expected_time']

            # Confidence level calculation
            confidence = self._calculate_combined_confidence(rule_based, ml_pred, final_prob)

            return {
                'prediction': prediction,
                'probability': final_prob,
                'expected_time': expected_time,
                'confidence': confidence,
                'details': details,
                'ml_confidence': ml_pred['confidence']
            }

        except Exception as e:
            logger.error(f"Error combining predictions: {str(e)}")
            return {
                'prediction': 'Tahmin hesaplanırken hata oluştu',
                'probability': 0.0,
                'expected_time': None,
                'confidence': 'düşük'
            }

    def _calculate_combined_confidence(self, rule_based: Dict, ml_pred: Dict, final_prob: float) -> str:
        """Birleşik güven seviyesini hesapla"""
        confidence_score = 0.0

        # ML güven seviyesi katkısı
        if ml_pred['confidence'] == 'yüksek':
            confidence_score += 0.4
        elif ml_pred['confidence'] == 'orta':
            confidence_score += 0.25
        else:
            confidence_score += 0.1

        # Olasılık bazlı katkı
        confidence_score += final_prob * 0.4

        # Veri kalitesi katkısı
        if 'data_quality' in ml_pred:
            confidence_score += ml_pred['data_quality'] * 0.2

        if confidence_score > 0.7:
            return 'yüksek'
        elif confidence_score > 0.5:
            return 'orta'
        else:
            return 'düşük'

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