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

    def extract_features(self, match_stats: Dict, events: List[Dict]) -> np.ndarray:
        """Maç istatistiklerinden özellik çıkarımı"""
        try:
            # Temel istatistikler
            home_stats = match_stats[0]['statistics']
            away_stats = match_stats[1]['statistics']

            features = []
            
            # Şut istatistikleri
            features.extend([
                float(home_stats[2]['value'] or 0),  # Home shots
                float(away_stats[2]['value'] or 0),  # Away shots
                float(home_stats[0]['value'] or 0),  # Home shots on target
                float(away_stats[0]['value'] or 0),  # Away shots on target
            ])

            # Top kontrolü
            home_possession = float(home_stats[9]['value'].strip('%')) if home_stats[9]['value'] else 50
            features.extend([
                home_possession / 100,
                (100 - home_possession) / 100
            ])

            # Tehlikeli ataklar
            features.extend([
                float(home_stats[13]['value'] or 0),  # Home dangerous attacks
                float(away_stats[13]['value'] or 0),  # Away dangerous attacks
            ])

            # Son olaylardan özellik çıkarımı
            recent_events = events[-10:] if events else []
            home_momentum = 0
            away_momentum = 0

            for event in recent_events:
                if event['team']['name'] == match_stats[0]['team']['name']:  # Home team
                    if event['type'] == 'Goal':
                        home_momentum += 0.3
                    elif event['type'] == 'Card':
                        home_momentum -= 0.1
                else:  # Away team
                    if event['type'] == 'Goal':
                        away_momentum += 0.3
                    elif event['type'] == 'Card':
                        away_momentum -= 0.1

            features.extend([home_momentum, away_momentum])

            # Normalize features
            features = np.array(features).reshape(1, -1)
            return self.scaler.fit_transform(features)

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return np.zeros((1, 10))  # Return zero features in case of error

    def predict_goals(self, match_stats: Dict, events: List[Dict]) -> Dict:
        """Gol olasılığını tahmin et"""
        try:
            features = self.extract_features(match_stats, events)
            
            # Model tahminleri
            prediction = self.model.predict(features)[0]
            
            # Tahmin sonuçlarını normalize et
            goal_prob = min(max(prediction, 0), 1)
            
            # Güven skorunu hesapla
            confidence_score = self._calculate_confidence(features, match_stats)
            
            return {
                'goal_probability': float(goal_prob),
                'confidence': confidence_score,
                'expected_time': self._predict_goal_time(events, goal_prob)
            }

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'goal_probability': 0.0,
                'confidence': 'düşük',
                'expected_time': None
            }

    def _calculate_confidence(self, features: np.ndarray, match_stats: Dict) -> str:
        """Tahmin güven seviyesini hesapla"""
        try:
            # Feature değerlerinin varyansını kontrol et
            feature_variance = np.var(features)
            
            # Veri kalitesini kontrol et
            data_quality = self._check_data_quality(match_stats)
            
            # Kombine güven skoru
            confidence_score = (feature_variance + data_quality) / 2
            
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
        """Veri kalitesini kontrol et"""
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
        """Gol zamanını tahmin et"""
        try:
            if not events:
                return None

            current_time = events[-1]['time']['elapsed']
            
            # Olasılığa göre zaman aralığı belirle
            if goal_prob > 0.7:
                time_range = (5, 15)  # Yüksek olasılık: 5-15 dakika
            elif goal_prob > 0.4:
                time_range = (10, 25)  # Orta olasılık: 10-25 dakika
            else:
                time_range = (15, 35)  # Düşük olasılık: 15-35 dakika

            predicted_time = current_time + np.random.randint(time_range[0], time_range[1])
            return min(90, predicted_time)

        except Exception as e:
            logger.error(f"Error predicting goal time: {str(e)}")
            return None
