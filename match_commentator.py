import os
from typing import Dict, List, Optional, Tuple
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
            
            # Mevcut dakikayı bul
            current_minute = events[-1]['time']['elapsed'] if events else 0
            
            # Son olayların analizi
            if events:
                logger.info("Processing recent events")
                try:
                    # Sadece mevcut dakikadan önceki olayları filtrele
                    valid_events = [event for event in events if event['time']['elapsed'] <= current_minute]
                    recent_events = valid_events[-3:]  # Son 3 olay
                    for event in recent_events:
                        if event['type'] == 'Goal':
                            commentary.append(f"⚽ {event['time']['elapsed']}. dakikada {event['team']['name']} golü buldu!")
                        elif event['type'] == 'Card':
                            commentary.append(f"🟨 {event['time']['elapsed']}. dakikada kart görüldü, oyun sertleşiyor.")
                except Exception as e:
                    logger.error(f"Error processing events: {str(e)}")
                    commentary.append("Maç olayları analiz edilirken bir hata oluştu.")

            # Skor analizi - olaylardan sonra yapılıyor
            home_score, away_score = score
            if home_score > away_score:
                score_diff = home_score - away_score
                if score_diff >= 3:
                    commentary.insert(0, "Ev sahibi takım maça tam hakimiyet kurmuş durumda.")
                else:
                    commentary.insert(0, "Ev sahibi takım önde, ancak maç hala dengeli.")
            elif away_score > home_score:
                score_diff = away_score - home_score
                if score_diff >= 3:
                    commentary.insert(0, "Deplasman takımı sahada üstünlüğü ele geçirmiş görünüyor.")
                else:
                    commentary.insert(0, "Deplasman takımı önde, fakat maç henüz bitmedi.")
            else:
                commentary.insert(0, "Şu an için skorlar eşit, her iki takım da üstünlük kurmaya çalışıyor.")

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
                            commentary.insert(1, f"Ev sahibi takım %{home_possession:.0f} top kontrolüyle oyunu yönlendiriyor.")
                        else:
                            commentary.insert(1, f"Deplasman takımı %{100-home_possession:.0f} top kontrolüyle oyuna hakim.")

                    # Şut analizi
                    home_shots = int(home_stats[2]['value'] or 0)
                    away_shots = int(away_stats[2]['value'] or 0)
                    if abs(home_shots - away_shots) > 3:
                        if home_shots > away_shots:
                            commentary.insert(1, f"Ev sahibi {home_shots} isabetli şutla rakibinden daha etkili.")
                        else:
                            commentary.insert(1, f"Deplasman {away_shots} isabetli şutla pozisyonları değerlendirmede daha başarılı.")
                except Exception as e:
                    logger.error(f"Error processing match statistics: {str(e)}")
                    commentary.append("İstatistik analizi yapılırken bir hata oluştu.")

            return " ".join(commentary)
        except Exception as e:
            logger.error(f"Error generating match commentary: {str(e)}")
            return "Maç yorumu oluşturulurken bir hata meydana geldi."

    def predict_next_goal(self, stats: Dict, events: List[Dict], historical_data: Dict) -> Dict:
        """Sonraki gol tahmini yap"""
        try:
            # MLPredictor'ı başlat
            predictor = MLPredictor()
            
            # Tahmin yap
            prediction = predictor.predict_goals(stats, events, historical_data)
            
            # Sonucu formatla
            result = {
                'prediction': f"Sonraki {prediction['predicted_goals']:.1f} gol",
                'probability': prediction['ensemble_confidence'],
                'confidence': 'yüksek' if prediction['ensemble_confidence'] > 0.7 else 'orta' if prediction['ensemble_confidence'] > 0.4 else 'düşük'
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Tahmin hatası: {str(e)}")
            return {
                'error': str(e)
            }

    def _analyze_betting_odds(self, betting_odds: Dict) -> Dict:
        """Bahis oranlarından detaylı analiz çıkar"""
        try:
            analysis = {}

            # Oranlardan olasılıkları hesapla
            total_prob = sum(1/float(odd) for odd in betting_odds.values())
            implied_probs = {k: (1/float(v))/total_prob for k, v in betting_odds.items()}

            # Market güvenilirliği
            market_efficiency = min(1.0, 1/total_prob)
            market_depth = min(1.0, len(betting_odds) / 10)

            # Bahis tiplerine göre analiz
            odds_by_type = {
                'match_result': {},
                'goals': {},
                'first_half': {}
            }

            for bet_type, odd in betting_odds.items():
                if bet_type in ['home', 'draw', 'away']:
                    odds_by_type['match_result'][bet_type] = float(odd)
                elif bet_type in ['over25', 'under25', 'btts']:
                    odds_by_type['goals'][bet_type] = float(odd)
                elif bet_type.startswith('first_half'):
                    odds_by_type['first_half'][bet_type] = float(odd)

            analysis = {
                'implied_probabilities': implied_probs,
                'market_quality': {
                    'efficiency': market_efficiency,
                    'depth': market_depth,
                    'confidence': (market_efficiency + market_depth) / 2
                },
                'odds_by_type': odds_by_type
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing betting odds: {str(e)}")
            return {}

    def _combine_all_predictions(self, ml_pred: Dict, rule_pred: Dict, 
                               odds_analysis: Optional[Dict], betting_odds: Optional[Dict]) -> Dict:
        """Tüm tahmin kaynaklarını birleştir"""
        try:
            predictions = {'predictions': {}}
            confidence_levels = ['yüksek', 'düşük']

            for level in confidence_levels:
                pred_data = {}

                # ML tahmin ağırlığı
                ml_weight = 0.6 if level == 'yüksek' else 0.4
                # Kural bazlı tahmin ağırlığı
                rule_weight = 0.4 if level == 'yüksek' else 0.6

                # Temel olasılık hesaplama
                base_prob = (
                    ml_pred.get('probability', 0.5) * ml_weight +
                    rule_pred.get('probability', 0.5) * rule_weight
                )

                # Bahis oranları varsa entegre et
                if odds_analysis and betting_odds:
                    market_prob = odds_analysis['implied_probabilities'].get('next_goal', 0.5)
                    final_prob = base_prob * 0.8 + market_prob * 0.2

                    # Tahmin nedenleri
                    reasons = []
                    if final_prob > 0.6:
                        reasons.append("Yüksek olasılıklı gol beklentisi")
                    if odds_analysis['market_quality']['confidence'] > 0.7:
                        reasons.append("Güçlü bahis market göstergeleri")

                    pred_data.update({
                        'probability': final_prob,
                        'reason': " & ".join(reasons) if reasons else "Standart analiz",
                        'odds_comparison': {
                            'market_odds': betting_odds,
                            'implied_probabilities': odds_analysis['implied_probabilities'],
                            'market_confidence': odds_analysis['market_quality']['confidence']
                        }
                    })
                else:
                    pred_data.update({
                        'probability': base_prob,
                        'reason': "Model ve kural bazlı tahmin"
                    })

                predictions['predictions'][level] = pred_data

            # En yüksek güvenli tahmini seç
            recommended = max(predictions['predictions'].items(),
                            key=lambda x: x[1]['probability'])

            result = {
                'prediction': self._generate_prediction_text(recommended[1]['probability']),
                'probability': recommended[1]['probability'],
                'confidence': recommended[0],
                'reason': recommended[1]['reason'],
                'predictions': predictions['predictions']
            }

            # Bahis oranları varsa ekle
            if betting_odds:
                result.update({
                    'betting_odds': betting_odds,
                    'odds_analysis': odds_analysis
                })

            return result

        except Exception as e:
            logger.error(f"Error combining predictions: {str(e)}")
            return {
                'prediction': 'Tahmin hesaplanamadı',
                'probability': 0.0,
                'confidence': 'düşük'
            }

    def _generate_prediction_text(self, probability: float) -> str:
        """Olasılığa göre tahmin metni oluştur"""
        if probability > 0.7:
            return 'Yüksek gol olasılığı'
        elif probability > 0.5:
            return 'Orta seviye gol olasılığı'
        elif probability > 0.3:
            return 'Düşük gol olasılığı'
        else:
            return 'Gol beklentisi çok düşük'

    def _calculate_rule_based_prediction(self, home_stats: Dict, away_stats: Dict, 
                                       events: List[Dict], match_stats: Dict) -> Dict:
        """Kural tabanlı tahmin hesapla"""
        try:
            # Momentum hesaplama
            home_momentum = self._calculate_team_momentum(home_stats, events, match_stats[0]['team']['name'], True)
            away_momentum = self._calculate_team_momentum(away_stats, events, match_stats[1]['team']['name'], False)

            # Son olayların analizi
            recent_events_impact = self._analyze_recent_events(events)

            # Temel olasılık hesaplama
            base_prob = (home_momentum + away_momentum) / 2

            # Son olayların etkisini ekle
            final_prob = base_prob * (1 + recent_events_impact)

            # Güven seviyesi hesaplama
            confidence = min(1.0, (base_prob + recent_events_impact) / 2)

            return {
                'probability': min(1.0, max(0.0, final_prob)),
                'confidence': confidence,
                'momentum': {
                    'home': home_momentum,
                    'away': away_momentum
                }
            }

        except Exception as e:
            logger.error(f"Error in rule-based prediction: {str(e)}")
            return {'probability': 0.5, 'confidence': 0.3}

    def _analyze_recent_events(self, events: List[Dict]) -> float:
        """Son olayların etkisini hesapla"""
        if not events:
            return 0.0

        impact = 0.0
        recent_events = events[-5:]  # Son 5 olay

        for event in recent_events:
            if event['type'] == 'Goal':
                impact += 0.2
            elif event['type'] == 'Card':
                impact += 0.1
            elif event['type'] == 'subst':
                impact += 0.05

        return min(0.5, impact)  # Maksimum 0.5 etki

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