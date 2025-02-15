import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PlayerAnalyzer:
    def __init__(self):
        """Oyuncu performans analizi sınıfı"""
        logger.info("Initializing Player Analyzer")

    def analyze_player_performance(self, player_stats: Dict) -> Dict:
        """Oyuncu performans metrikleri hesaplama"""
        try:
            if not player_stats:
                return self._create_empty_performance()

            # Temel performans metrikleri
            performance = {
                'scoring_ability': self._calculate_scoring_ability(player_stats),
                'passing_efficiency': self._calculate_passing_metrics(player_stats),
                'defensive_strength': self._calculate_defensive_metrics(player_stats),
                'form_trend': self._calculate_form_trend(player_stats),
                'match_influence': self._calculate_match_influence(player_stats),
                'opponent_adjusted_rating': self._calculate_opponent_adjusted_rating(player_stats)
            }

            # Oyuncu rolüne göre ağırlıklandırılmış skor
            performance['overall_rating'] = self._calculate_weighted_rating(
                performance, player_stats.get('position', '')
            )

            return performance

        except Exception as e:
            logger.error(f"Error analyzing player performance: {str(e)}")
            return self._create_empty_performance()

    def _calculate_opponent_adjusted_rating(self, stats: Dict) -> float:
        """Rakip seviyesine göre düzeltilmiş performans hesaplama"""
        try:
            base_rating = float(stats.get('games', {}).get('rating', 0) or 0)
            opponent_strength = float(stats.get('opponent_strength', 0.5) or 0.5)

            # Rakip gücüne göre performansı ağırlıklandır
            adjusted_rating = base_rating * (1 + (opponent_strength - 0.5))

            return min(1.0, max(0.0, adjusted_rating / 10))  # 0-1 aralığına normalize et

        except Exception as e:
            logger.error(f"Error calculating opponent adjusted rating: {str(e)}")
            return 0.0

    def _calculate_scoring_ability(self, stats: Dict) -> float:
        """Gol atma yeteneği hesaplama"""
        try:
            goals = float(stats.get('goals', {}).get('total', 0) or 0)
            shots = float(stats.get('shots', {}).get('total', 0) or 1)
            games = float(stats.get('games', {}).get('appearences', 0) or 1)
            minutes = float(stats.get('games', {}).get('minutes', 0) or 1)

            scoring_rate = goals / games if games > 0 else 0
            shot_accuracy = float(stats.get('shots', {}).get('on', 0) or 0) / shots if shots > 0 else 0
            minutes_per_goal = minutes / max(goals, 1)

            # Dakika başına gol ve şut isabet oranını hesaba kat
            weighted_score = (
                scoring_rate * 0.4 +
                shot_accuracy * 0.3 +
                min(1.0, 90 / max(minutes_per_goal, 1)) * 0.3
            )

            return min(1.0, weighted_score)

        except Exception as e:
            logger.error(f"Error calculating scoring ability: {str(e)}")
            return 0.0

    def _calculate_passing_metrics(self, stats: Dict) -> float:
        """Pas performansı metrikleri"""
        try:
            total_passes = float(stats.get('passes', {}).get('total', 0) or 1)
            accurate_passes = float(stats.get('passes', {}).get('accuracy', 0) or 0)
            key_passes = float(stats.get('passes', {}).get('key', 0) or 0)
            assists = float(stats.get('goals', {}).get('assists', 0) or 0)
            games = float(stats.get('games', {}).get('appearences', 0) or 1)

            pass_accuracy = accurate_passes / total_passes if total_passes > 0 else 0
            key_pass_rate = key_passes / games
            assist_rate = assists / games

            return (pass_accuracy * 0.4 + key_pass_rate * 0.3 + assist_rate * 0.3)

        except Exception as e:
            logger.error(f"Error calculating passing metrics: {str(e)}")
            return 0.0

    def _calculate_defensive_metrics(self, stats: Dict) -> float:
        """Savunma performansı metrikleri"""
        try:
            games = float(stats.get('games', {}).get('appearences', 0) or 1)
            tackles = float(stats.get('tackles', {}).get('total', 0) or 0)
            interceptions = float(stats.get('tackles', {}).get('interceptions', 0) or 0)
            duels_won = float(stats.get('duels', {}).get('won', 0) or 0)
            duels_total = float(stats.get('duels', {}).get('total', 0) or 1)

            tackle_rate = tackles / games
            interception_rate = interceptions / games
            duel_success = duels_won / duels_total if duels_total > 0 else 0

            return (tackle_rate * 0.4 + interception_rate * 0.3 + duel_success * 0.3)

        except Exception as e:
            logger.error(f"Error calculating defensive metrics: {str(e)}")
            return 0.0

    def _calculate_form_trend(self, stats: Dict) -> float:
        """Form trendi analizi"""
        try:
            recent_ratings = stats.get('games', {}).get('rating', []) or []
            if not recent_ratings:
                return 0.0

            # Son 5 maçın ağırlıklı ortalaması
            weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])[:len(recent_ratings)]
            weights = weights / weights.sum()

            ratings = np.array(recent_ratings[-5:])
            base_trend = float(np.sum(weights * ratings) / 10)  # 0-1 aralığına normalize et

            # Form momentum hesapla
            if len(ratings) >= 2:
                momentum = (ratings[-1] - ratings[0]) / len(ratings)
                momentum_factor = max(-0.2, min(0.2, momentum / 10))  # ±0.2 sınırlama
                base_trend = max(0.0, min(1.0, base_trend + momentum_factor))

            return base_trend

        except Exception as e:
            logger.error(f"Error calculating form trend: {str(e)}")
            return 0.0

    def _calculate_match_influence(self, stats: Dict) -> float:
        """Maç etki skorunu hesapla"""
        try:
            games = float(stats.get('games', {}).get('appearences', 0) or 1)
            minutes = float(stats.get('games', {}).get('minutes', 0) or 0)
            goals = float(stats.get('goals', {}).get('total', 0) or 0)
            assists = float(stats.get('goals', {}).get('assists', 0) or 0)
            key_passes = float(stats.get('passes', {}).get('key', 0) or 0)
            duels_won = float(stats.get('duels', {}).get('won', 0) or 0)
            duels_total = float(stats.get('duels', {}).get('total', 0) or 1)

            minutes_per_game = minutes / games if games > 0 else 0
            goal_involvement = (goals + assists) / games
            key_pass_rate = key_passes / games
            duel_success = duels_won / duels_total if duels_total > 0 else 0

            influence_score = (
                (minutes_per_game / 90) * 0.2 +
                goal_involvement * 0.3 +
                key_pass_rate * 0.2 +
                duel_success * 0.3
            )

            return max(0, min(1, influence_score))

        except Exception as e:
            logger.error(f"Error calculating match influence: {str(e)}")
            return 0.0

    def _calculate_weighted_rating(self, performance: Dict, position: str) -> float:
        """Pozisyona göre ağırlıklandırılmış performans skoru"""
        try:
            weights = {
                'Attacker': {
                    'scoring_ability': 0.5,
                    'passing_efficiency': 0.2,
                    'defensive_strength': 0.1,
                    'form_trend': 0.1,
                    'match_influence': 0.1,
                    'opponent_adjusted_rating': 0.1
                },
                'Midfielder': {
                    'scoring_ability': 0.2,
                    'passing_efficiency': 0.4,
                    'defensive_strength': 0.2,
                    'form_trend': 0.1,
                    'match_influence': 0.1,
                    'opponent_adjusted_rating': 0.1
                },
                'Defender': {
                    'scoring_ability': 0.1,
                    'passing_efficiency': 0.2,
                    'defensive_strength': 0.5,
                    'form_trend': 0.1,
                    'match_influence': 0.1,
                    'opponent_adjusted_rating': 0.1
                }
            }

            # Pozisyon tipini belirle
            if 'forward' in position.lower() or 'striker' in position.lower():
                pos_type = 'Attacker'
            elif 'midfielder' in position.lower():
                pos_type = 'Midfielder'
            elif 'defender' in position.lower() or 'back' in position.lower():
                pos_type = 'Defender'
            else:
                pos_type = 'Midfielder'  # Varsayılan

            # Ağırlıklı skor hesapla
            weighted_score = sum(
                performance[metric] * weight
                for metric, weight in weights[pos_type].items()
            )

            return weighted_score

        except Exception as e:
            logger.error(f"Error calculating weighted rating: {str(e)}")
            return 0.0

    def _create_empty_performance(self) -> Dict:
        """Boş performans verisi oluştur"""
        return {
            'scoring_ability': 0.0,
            'passing_efficiency': 0.0,
            'defensive_strength': 0.0,
            'form_trend': 0.0,
            'match_influence': 0.0,
            'opponent_adjusted_rating': 0.0,
            'overall_rating': 0.0
        }

    def get_performance_summary(self, performance: Dict) -> str:
        """Performans özeti oluştur"""
        try:
            if not performance:
                return "Performans verisi bulunamadı."

            rating = performance['overall_rating']
            form_trend = performance.get('form_trend', 0)
            opponent_rating = performance.get('opponent_adjusted_rating', 0)

            # Genel durumu belirle
            if rating > 0.8:
                form = "Çok iyi formda"
            elif rating > 0.6:
                form = "İyi formda"
            elif rating > 0.4:
                form = "Ortalama formda"
            else:
                form = "Form düşüşünde"

            # Trend durumunu ekle
            trend_text = ""
            if form_trend > 0.6:
                trend_text = ", yükselen performans"
            elif form_trend < 0.4:
                trend_text = ", düşen performans"

            # Rakip seviyesine göre düzeltilmiş değerlendirme
            opponent_text = ""
            if opponent_rating > rating + 0.2:
                opponent_text = " (güçlü rakiplere karşı etkili)"
            elif opponent_rating < rating - 0.2:
                opponent_text = " (zayıf rakiplere karşı)"

            return f"{form}{trend_text}{opponent_text} (Rating: {rating:.2f})"

        except Exception as e:
            logger.error(f"Error creating performance summary: {str(e)}")
            return "Performans analizi yapılamadı."