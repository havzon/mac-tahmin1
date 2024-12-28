import logging
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulationEngine:
    def __init__(self):
        """Tahmin simülasyonu motoru"""
        self.simulations = {}
        self.simulation_results = {}
        
    def create_simulation(self, match_id: str, initial_state: Dict) -> str:
        """Yeni bir simülasyon oluştur"""
        try:
            simulation_id = f"sim_{match_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.simulations[simulation_id] = {
                'match_id': match_id,
                'initial_state': initial_state,
                'predictions': [],
                'created_at': datetime.now()
            }
            return simulation_id
        except Exception as e:
            logger.error(f"Simülasyon oluşturma hatası: {str(e)}")
            return None

    def add_prediction(self, simulation_id: str, prediction: Dict) -> bool:
        """Simülasyona tahmin ekle"""
        try:
            if simulation_id not in self.simulations:
                return False
                
            self.simulations[simulation_id]['predictions'].append({
                'timestamp': datetime.now(),
                'prediction': prediction
            })
            return True
        except Exception as e:
            logger.error(f"Tahmin ekleme hatası: {str(e)}")
            return False

    def calculate_simulation_score(self, simulation_id: str, actual_result: Dict) -> Dict:
        """Simülasyon skorunu hesapla"""
        try:
            if simulation_id not in self.simulations:
                return None

            simulation = self.simulations[simulation_id]
            predictions = simulation['predictions']

            if not predictions:
                return {
                    'score': 0,
                    'accuracy': 0,
                    'details': "Tahmin bulunamadı"
                }

            # Tahmin doğruluğunu hesapla
            accuracy_scores = []
            for pred in predictions:
                pred_value = pred['prediction'].get('result', 0)
                actual_value = actual_result.get('result', 0)
                accuracy = 1 - abs(pred_value - actual_value)
                accuracy_scores.append(accuracy)

            avg_accuracy = np.mean(accuracy_scores)
            
            # Sonuç detayları
            result = {
                'score': int(avg_accuracy * 100),
                'accuracy': avg_accuracy,
                'prediction_count': len(predictions),
                'details': self._generate_simulation_feedback(avg_accuracy)
            }

            self.simulation_results[simulation_id] = result
            return result

        except Exception as e:
            logger.error(f"Simülasyon skoru hesaplama hatası: {str(e)}")
            return None

    def get_simulation_stats(self, simulation_id: str) -> Dict:
        """Simülasyon istatistiklerini getir"""
        try:
            if simulation_id not in self.simulations:
                return None

            simulation = self.simulations[simulation_id]
            predictions = simulation['predictions']

            if not predictions:
                return {
                    'total_predictions': 0,
                    'average_confidence': 0,
                    'prediction_timeline': []
                }

            # İstatistikleri hesapla
            confidences = [p['prediction'].get('confidence', 0) for p in predictions]
            timeline = [{'time': p['timestamp'], 'value': p['prediction'].get('result', 0)} 
                       for p in predictions]

            return {
                'total_predictions': len(predictions),
                'average_confidence': np.mean(confidences),
                'prediction_timeline': timeline,
                'last_prediction': predictions[-1]['prediction'] if predictions else None
            }

        except Exception as e:
            logger.error(f"Simülasyon istatistikleri alma hatası: {str(e)}")
            return None

    def _generate_simulation_feedback(self, accuracy: float) -> str:
        """Simülasyon geri bildirimi oluştur"""
        if accuracy >= 0.9:
            return "Mükemmel tahmin performansı! Analiziniz çok başarılı."
        elif accuracy >= 0.7:
            return "İyi performans! Tahminleriniz genelde isabetli."
        elif accuracy >= 0.5:
            return "Ortalama performans. Gelişim alanları mevcut."
        else:
            return "Tahmin performansınızı geliştirmek için daha fazla analiz yapmanız önerilir."

    def get_learning_suggestions(self, simulation_id: str) -> List[str]:
        """Öğrenme önerileri oluştur"""
        try:
            if simulation_id not in self.simulations:
                return []

            stats = self.get_simulation_stats(simulation_id)
            suggestions = []

            if stats['average_confidence'] < 0.6:
                suggestions.append("Tahminlerinizde daha güvenli olmak için maç istatistiklerini detaylı inceleyin.")
            
            if stats['total_predictions'] < 3:
                suggestions.append("Daha fazla tahmin yaparak deneyiminizi artırın.")

            if simulation_id in self.simulation_results:
                result = self.simulation_results[simulation_id]
                if result['accuracy'] < 0.5:
                    suggestions.append("Tahmin doğruluğunuzu artırmak için takım form analizlerine odaklanın.")

            return suggestions

        except Exception as e:
            logger.error(f"Öğrenme önerileri oluşturma hatası: {str(e)}")
            return []
