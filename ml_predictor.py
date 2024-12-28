import numpy as np
import xgboost as xgb
import logging
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self):
        """ML tabanlı gol tahmin sistemi"""
        logger.info("Initializing ML Predictor")
        self.model = None
        self.scaler = StandardScaler()
        self._initialize_model()

    def _initialize_model(self):
        """XGBoost modelini başlat"""
        try:
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                objective='reg:squarederror'
            )
            logger.info("ML model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ML model: {str(e)}")
            raise

    def extract_features(self, match_stats: Dict, events: List[Dict], historical_data: Optional[Dict] = None) -> np.ndarray:
        """Maç istatistiklerinden gelişmiş özellik çıkarımı"""
        try:
            home_stats = match_stats[0]['statistics']
            away_stats = match_stats[1]['statistics']
            home_team = match_stats[0]['team']['name']
            away_team = match_stats[1]['team']['name']

            features = []

            # Temel istatistikler ve özelleştirilmiş ağırlıklar
            team_weights = self._calculate_team_weights(home_team, away_team, historical_data)

            # Şut ve gol şansı metrikleri
            shot_metrics = self._calculate_shot_metrics(home_stats, away_stats)
            features.extend([
                shot_metrics['home_shot_efficiency'] * team_weights['home_attack'],
                shot_metrics['away_shot_efficiency'] * team_weights['away_attack'],
                shot_metrics['home_conversion_rate'] * team_weights['home_finish'],
                shot_metrics['away_conversion_rate'] * team_weights['away_finish']
            ])

            # Taktiksel metrikler
            tactical_metrics = self._calculate_tactical_metrics(home_stats, away_stats)
            features.extend([
                tactical_metrics['home_possession_effectiveness'] * team_weights['home_control'],
                tactical_metrics['away_possession_effectiveness'] * team_weights['away_control'],
                tactical_metrics['home_pressing_success'] * team_weights['home_press'],
                tactical_metrics['away_pressing_success'] * team_weights['away_press']
            ])

            # Momentum ve form analizi
            momentum_metrics = self._calculate_momentum_metrics(events, home_team, away_team)
            features.extend([
                momentum_metrics['home_momentum'] * team_weights['home_momentum'],
                momentum_metrics['away_momentum'] * team_weights['away_momentum'],
                momentum_metrics['home_energy'],
                momentum_metrics['away_energy']
            ])

            # Maç durumu ve zaman bazlı faktörler
            time_metrics = self._calculate_time_based_metrics(events)
            features.extend([
                time_metrics['game_intensity'],
                time_metrics['time_pressure'],
                time_metrics['goal_probability_modifier']
            ])

            # Normalize features
            features = np.array(features).reshape(1, -1)
            return self.scaler.fit_transform(features)

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return np.zeros((1, 15))  # Genişletilmiş özellik vektörü

    def _calculate_team_weights(self, home_team: str, away_team: str, historical_data: Optional[Dict]) -> Dict:
        """Takım bazlı ağırlıkları hesapla"""
        try:
            # Varsayılan ağırlıklar
            weights = {
                'home_attack': 1.0, 'away_attack': 1.0,
                'home_finish': 1.0, 'away_finish': 1.0,
                'home_control': 1.0, 'away_control': 1.0,
                'home_press': 1.0, 'away_press': 1.0,
                'home_momentum': 1.0, 'away_momentum': 1.0
            }

            if historical_data:
                # Geçmiş performansa dayalı ağırlık ayarlaması
                home_form = historical_data.get(home_team, {}).get('form', 0.5)
                away_form = historical_data.get(away_team, {}).get('form', 0.5)

                weights['home_attack'] *= (1 + home_form)
                weights['away_attack'] *= (1 + away_form)
                weights['home_finish'] *= (1 + home_form * 0.5)
                weights['away_finish'] *= (1 + away_form * 0.5)

            return weights

        except Exception as e:
            logger.error(f"Error calculating team weights: {str(e)}")
            return dict.fromkeys(weights.keys(), 1.0)

    def _calculate_shot_metrics(self, home_stats: List[Dict], away_stats: List[Dict]) -> Dict:
        """Şut ve gol şansı metriklerini hesapla"""
        try:
            # Ev sahibi metrikleri
            home_shots = float(home_stats[2]['value'] or 0)
            home_shots_on_target = float(home_stats[0]['value'] or 0)
            home_dangerous_attacks = float(home_stats[13]['value'] or 0)

            # Deplasman metrikleri
            away_shots = float(away_stats[2]['value'] or 0)
            away_shots_on_target = float(away_stats[0]['value'] or 0)
            away_dangerous_attacks = float(away_stats[13]['value'] or 0)

            return {
                'home_shot_efficiency': home_shots_on_target / max(home_shots, 1),
                'away_shot_efficiency': away_shots_on_target / max(away_shots, 1),
                'home_conversion_rate': home_dangerous_attacks / max(home_shots * 2, 1),
                'away_conversion_rate': away_dangerous_attacks / max(away_shots * 2, 1)
            }

        except Exception as e:
            logger.error(f"Error calculating shot metrics: {str(e)}")
            return dict.fromkeys(['home_shot_efficiency', 'away_shot_efficiency',
                                'home_conversion_rate', 'away_conversion_rate'], 0.0)

    def _calculate_tactical_metrics(self, home_stats: List[Dict], away_stats: List[Dict]) -> Dict:
        """Taktiksel metrikleri hesapla"""
        try:
            # Possession ve kontrol metrikleri
            home_possession = float(home_stats[9]['value'].strip('%')) / 100 if home_stats[9]['value'] else 0.5
            away_possession = 1 - home_possession

            # Pressing ve agresiflik
            home_fouls = float(home_stats[7]['value'] or 0)
            away_fouls = float(away_stats[7]['value'] or 0)

            # Taktik etkinliği
            home_passes = float(home_stats[12]['value'] or 0)
            away_passes = float(away_stats[12]['value'] or 0)

            return {
                'home_possession_effectiveness': home_possession * (home_passes / max(home_passes + away_passes, 1)),
                'away_possession_effectiveness': away_possession * (away_passes / max(home_passes + away_passes, 1)),
                'home_pressing_success': 1 - (away_passes / max(home_passes + away_passes, 1)),
                'away_pressing_success': 1 - (home_passes / max(home_passes + away_passes, 1))
            }

        except Exception as e:
            logger.error(f"Error calculating tactical metrics: {str(e)}")
            return dict.fromkeys(['home_possession_effectiveness', 'away_possession_effectiveness',
                                'home_pressing_success', 'away_pressing_success'], 0.0)

    def _calculate_momentum_metrics(self, events: List[Dict], home_team: str, away_team: str) -> Dict:
        """Momentum ve enerji metriklerini hesapla"""
        try:
            metrics = {
                'home_momentum': 0.0, 'away_momentum': 0.0,
                'home_energy': 1.0, 'away_energy': 1.0
            }

            if not events:
                return metrics

            # Son olayların analizi
            recent_events = events[-10:]  # Son 10 olay
            for event in recent_events:
                event_team = event['team']['name']
                event_time = event['time']['elapsed']

                # Olay ağırlıkları
                weights = {
                    'Goal': 0.3,
                    'Card': -0.1,
                    'subst': 0.05,
                    'Var': 0.05,
                    'Shot': 0.1
                }

                # Momentum hesaplama
                event_weight = weights.get(event['type'], 0.0)
                if event_team == home_team:
                    metrics['home_momentum'] += event_weight
                    metrics['home_energy'] *= (1 - event_time/90 * 0.3)  # Enerji düşüşü
                else:
                    metrics['away_momentum'] += event_weight
                    metrics['away_energy'] *= (1 - event_time/90 * 0.3)

            # Normalize momentum
            total_momentum = abs(metrics['home_momentum']) + abs(metrics['away_momentum'])
            if total_momentum > 0:
                metrics['home_momentum'] /= total_momentum
                metrics['away_momentum'] /= total_momentum

            return metrics

        except Exception as e:
            logger.error(f"Error calculating momentum metrics: {str(e)}")
            return dict.fromkeys(['home_momentum', 'away_momentum', 'home_energy', 'away_energy'], 0.0)

    def _calculate_time_based_metrics(self, events: List[Dict]) -> Dict:
        """Zaman bazlı metrikleri hesapla"""
        try:
            if not events:
                return {'game_intensity': 0.5, 'time_pressure': 0.0, 'goal_probability_modifier': 1.0}

            current_time = events[-1]['time']['elapsed']

            # Oyun yoğunluğu
            recent_events = len([e for e in events if e['time']['elapsed'] > max(0, current_time - 15)])
            game_intensity = min(1.0, recent_events / 10)

            # Zaman baskısı
            time_pressure = min(1.0, max(0.0, (current_time - 60) / 30)) if current_time > 60 else 0.0

            # Gol olasılığı modifiyesi
            if current_time >= 85:
                goal_prob_mod = 1.5  # Son dakika baskısı
            elif current_time >= 70:
                goal_prob_mod = 1.2  # Oyun sonu yaklaşıyor
            else:
                goal_prob_mod = 1.0

            return {
                'game_intensity': game_intensity,
                'time_pressure': time_pressure,
                'goal_probability_modifier': goal_prob_mod
            }

        except Exception as e:
            logger.error(f"Error calculating time-based metrics: {str(e)}")
            return {'game_intensity': 0.5, 'time_pressure': 0.0, 'goal_probability_modifier': 1.0}

    def predict_goals(self, match_stats: Dict, events: List[Dict]) -> Dict:
        """Geliştirilmiş gol olasılığı tahmini"""
        try:
            features = self.extract_features(match_stats, events)

            # Model tahminleri
            base_prediction = self.model.predict(features)[0]

            # Zaman bazlı faktörler
            time_metrics = self._calculate_time_based_metrics(events)

            # Tahmin ayarlaması
            adjusted_prediction = base_prediction * time_metrics['goal_probability_modifier']

            # Güven skoru hesaplama
            confidence_score = self._calculate_confidence(features, match_stats)

            # Tahmini zaman hesaplama
            expected_time = self._predict_goal_time(events, adjusted_prediction)

            return {
                'goal_probability': max(0, min(1, adjusted_prediction)),
                'confidence': confidence_score,
                'expected_time': expected_time,
                'intensity': time_metrics['game_intensity'],
                'pressure': time_metrics['time_pressure']
            }

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'goal_probability': 0.0,
                'confidence': 'düşük',
                'expected_time': None,
                'intensity': 0.0,
                'pressure': 0.0
            }

    def _calculate_confidence(self, features: np.ndarray, match_stats: Dict) -> str:
        """Geliştirilmiş güven seviyesi hesaplama"""
        try:
            # Feature değerlerinin varyansı
            feature_variance = np.var(features)

            # Veri kalitesi kontrolü
            data_quality = self._check_data_quality(match_stats)

            # Güven skoru hesaplama
            confidence_score = (feature_variance * 0.4 + data_quality * 0.6)

            if confidence_score > 0.7:
                return 'yüksek'
            elif confidence_score > 0.4:
                return 'orta'
            else:
                return 'düşük'

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 'düşük'

    def _check_data_quality(self, match_stats: Dict) -> float:
        """Veri kalitesi kontrolü"""
        try:
            total_fields = 0
            valid_fields = 0

            for team_stats in match_stats:
                for stat in team_stats['statistics']:
                    total_fields += 1
                    if stat['value'] is not None and stat['value'] != '':
                        valid_fields += 1

            return valid_fields / max(total_fields, 1)

        except Exception as e:
            logger.error(f"Error checking data quality: {str(e)}")
            return 0.0

    def _predict_goal_time(self, events: List[Dict], goal_prob: float) -> Optional[int]:
        """Geliştirilmiş gol zamanı tahmini"""
        try:
            if not events:
                return None

            current_time = events[-1]['time']['elapsed']

            # Oyun temposu ve yoğunluğu
            time_metrics = self._calculate_time_based_metrics(events)
            game_intensity = time_metrics['game_intensity']

            # Olasılığa göre zaman aralığı belirleme
            if goal_prob > 0.7:
                base_range = (5, 15)  # Yüksek olasılık: yakın zamanda
            elif goal_prob > 0.4:
                base_range = (10, 25)  # Orta olasılık
            else:
                base_range = (15, 35)  # Düşük olasılık: daha uzun süre

            # Tempo bazlı ayarlama
            min_time = max(1, int(base_range[0] * (1 - game_intensity * 0.3)))
            max_time = max(min_time + 5, int(base_range[1] * (1 - game_intensity * 0.3)))

            predicted_time = current_time + np.random.randint(min_time, max_time)
            return min(90, predicted_time)

        except Exception as e:
            logger.error(f"Error predicting goal time: {str(e)}")
            return None