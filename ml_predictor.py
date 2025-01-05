import numpy as np
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
import logging

class MLPredictor:
    def __init__(self):
        """MLPredictor sınıfını başlat"""
        self.logger = logging.getLogger(__name__)
        self.xgb_model1 = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        self.xgb_model2 = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6
        )

    def create_training_data(self, num_samples: int = 100, real_data: Optional[List[Dict]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Eğitim verisi oluştur"""
        try:
            if real_data:
                # Gerçek veri kullanılıyorsa, özellikleri ve hedef değişkeni çıkar
                X = np.array([self._extract_features_from_real_data(data) for data in real_data])
                y = np.array([data['goals'] for data in real_data])
            else:
                # Gerçek veri yoksa, rastgele veri üret
                X = np.random.rand(num_samples, 200)
                y = np.random.poisson(lam=2.5, size=num_samples)
            
            return X, y

        except Exception as e:
            self.logger.error(f"Error creating training data: {str(e)}")
            return np.array([]), np.array([])

    def _extract_features_from_real_data(self, data: Dict) -> np.ndarray:
        """Gerçek veriden özellikler çıkar"""
        try:
            features = []

            # Maç istatistiklerinden özellikler
            features.append(data.get('shots_on_target', 0))
            features.append(data.get('total_shots', 0))
            features.append(data.get('possession', 50))
            features.append(data.get('corners', 0))
            features.append(data.get('fouls', 0))
            features.append(data.get('yellow_cards', 0))
            features.append(data.get('red_cards', 0))
            
            # Geçmiş verilerden özellikler
            features.append(data.get('avg_goals_scored', 0))
            features.append(data.get('avg_goals_conceded', 0))
            features.append(data.get('win_streak', 0))
            features.append(data.get('clean_sheets', 0))
            
            return np.array(features)

        except Exception as e:
            self.logger.error(f"Error extracting features from real data: {str(e)}")
            return np.zeros(11)  # Varsayılan özellik sayısı

    def _extract_enhanced_features(self, match_stats: Dict, events: List[Dict], historical_data: Dict) -> np.ndarray:
        """Gelişmiş özellik çıkarımı"""
        try:
            features = []

            # Mevcut maç istatistiklerinden özellikler
            for team_stats in match_stats.values():
                team_features = []
                for stat in team_stats['statistics']:
                    if stat['type'] in ['Total Shots', 'Corner Kicks', 'Yellow Cards', 'Ball Possession']:
                        value = stat['value']
                        if isinstance(value, str):
                            if '%' in value:
                                value = float(value.strip('%'))
                            else:
                                value = float(value)
                        team_features.append(value)
                features.extend(team_features)
            
            # Olay verilerinden özellikler
            event_counts = {
                'Goal': 0, 'Shot': 0, 'Corner': 0, 'Card': 0
            }
            for event in events:
                event_type = event['type']
                if event_type in event_counts:
                    event_counts[event_type] += 1
            features.extend(list(event_counts.values()))
            
            # Geçmiş verilerden özellikler
            for match in historical_data['recent_matches']:
                match_features = [
                    match.get('goals_scored', 0),
                    match.get('goals_conceded', 0),
                    match.get('possession', 50),
                    match.get('shots_on_target', 0),
                    match.get('shots_total', 0),
                    match.get('corners', 0),
                    match.get('yellow_cards', 0),
                    match.get('red_cards', 0),
                    match.get('team_rating', 0),
                    match.get('first_half_goals', 0),
                    match.get('second_half_goals', 0),
                    match.get('late_goals', 0),
                    match.get('win_streak', 0),
                    match.get('clean_sheets', 0),
                    match.get('formation_success_rate', 0),
                    match.get('counter_attack_goals', 0),
                    match.get('set_piece_goals', 0),
                    match.get('possession_score', 0),
                    match.get('defensive_rating', 0)
                ]
                features.extend(match_features)
            
            # Eksik özellikleri 0 ile doldur
            while len(features) < 200:
                features.append(0)
                
            # Fazla özellikleri kes
            features = features[:200]
            
            return np.array(features).reshape(1, -1)

        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return np.zeros((1, 200))

    def predict_goals(self, match_stats: Dict, events: List[Dict], historical_data: Dict) -> Dict:
        """Gol tahmini yap"""
        try:
            # Özellikleri çıkar
            features = self._extract_enhanced_features(match_stats, events, historical_data)
            
            # Her iki modelle tahmin yap
            pred1 = self.xgb_model1.predict(features)[0]
            pred2 = self.xgb_model2.predict(features)[0]
            
            # Tahminleri birleştir (ağırlıklı ortalama)
            ensemble_pred = 0.6 * pred1 + 0.4 * pred2
            
            # Güven skorlarını hesapla
            confidence1 = np.clip(1 - abs(pred1 - ensemble_pred) / ensemble_pred, 0.5, 1)
            confidence2 = np.clip(1 - abs(pred2 - ensemble_pred) / ensemble_pred, 0.5, 1)
            ensemble_confidence = (confidence1 + confidence2) / 2

            return {
                'predicted_goals': ensemble_pred,
                'model1_confidence': confidence1,
                'model2_confidence': confidence2,
                'ensemble_confidence': ensemble_confidence
            }

        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return {
                'predicted_goals': 0,
                'model1_confidence': 0,
                'model2_confidence': 0,
                'ensemble_confidence': 0
            }

    def train_models(self, training_data: Tuple[np.ndarray, np.ndarray]):
        """Modelleri eğit"""
        try:
            X, y = training_data
            self.xgb_model1.fit(X, y)
            self.xgb_model2.fit(X, y)
            self.logger.info("Modeller başarıyla eğitildi")
        except Exception as e:
            self.logger.error(f"Modeller eğitilirken hata oluştu: {str(e)}")