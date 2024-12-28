import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    def __init__(self):
        """Takım ve oyuncu performans analiz sistemi"""
        logger.info("Initializing PerformanceAnalyzer")
        self.initialized = True

    def analyze_team_performance(self, match_stats: Dict, events: List[Dict]) -> Dict:
        """Takım performans analizi yap"""
        try:
            if not match_stats or len(match_stats) < 2:
                return self._create_empty_analysis()

            home_stats = match_stats[0]['statistics']
            away_stats = match_stats[1]['statistics']
            
            # Temel metrikler
            home_metrics = self._calculate_team_metrics(home_stats, True)
            away_metrics = self._calculate_team_metrics(away_stats, False)
            
            # Trend analizi
            home_trend = self._analyze_performance_trend(events, match_stats[0]['team']['name'])
            away_trend = self._analyze_performance_trend(events, match_stats[1]['team']['name'])
            
            return {
                'home_team': {
                    'metrics': home_metrics,
                    'trend': home_trend,
                    'performance_score': self._calculate_performance_score(home_metrics, home_trend),
                    'strengths': self._identify_team_strengths(home_metrics),
                    'areas_to_improve': self._identify_improvement_areas(home_metrics)
                },
                'away_team': {
                    'metrics': away_metrics,
                    'trend': away_trend,
                    'performance_score': self._calculate_performance_score(away_metrics, away_trend),
                    'strengths': self._identify_team_strengths(away_metrics),
                    'areas_to_improve': self._identify_improvement_areas(away_metrics)
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing team performance: {str(e)}")
            return self._create_empty_analysis()

    def analyze_player_performance(self, events: List[Dict]) -> Dict:
        """Oyuncu performans analizi yap"""
        try:
            player_stats = {}
            
            for event in events:
                if 'player' not in event:
                    continue
                    
                player_id = event['player']['id']
                player_name = event['player']['name']
                team_name = event['team']['name']
                
                if player_id not in player_stats:
                    player_stats[player_id] = {
                        'name': player_name,
                        'team': team_name,
                        'events': [],
                        'impact_score': 0,
                        'key_moments': []
                    }
                
                # Olay analizi
                impact = self._calculate_event_impact(event)
                player_stats[player_id]['impact_score'] += impact
                
                if impact > 0.5:  # Önemli anlar
                    player_stats[player_id]['key_moments'].append({
                        'time': event['time']['elapsed'],
                        'event_type': event['type'],
                        'impact': impact
                    })
                
                player_stats[player_id]['events'].append(event)
            
            # Performans değerlendirmesi
            return self._evaluate_player_performances(player_stats)
            
        except Exception as e:
            logger.error(f"Error analyzing player performance: {str(e)}")
            return {}

    def _calculate_team_metrics(self, stats: List[Dict], is_home: bool) -> Dict:
        """Takım metriklerini hesapla"""
        try:
            metrics = {
                'offensive_efficiency': 0.0,
                'defensive_stability': 0.0,
                'possession_control': 0.0,
                'pressing_intensity': 0.0,
                'transition_speed': 0.0
            }
            
            # Hücum etkinliği
            shots = int(stats[2]['value'] or 0)
            shots_on_target = int(stats[0]['value'] or 0)
            metrics['offensive_efficiency'] = (shots_on_target / max(shots, 1)) * 0.7 + \
                                           (shots / 10) * 0.3
            
            # Savunma istikrarı
            tackles = int(stats[7]['value'] or 0)
            interceptions = int(stats[8]['value'] or 0)
            metrics['defensive_stability'] = (tackles / 20) * 0.5 + \
                                          (interceptions / 10) * 0.5
            
            # Top kontrolü
            possession = float(stats[9]['value'].strip('%')) if stats[9]['value'] else 50
            metrics['possession_control'] = possession / 100
            
            # Pressing yoğunluğu
            fouls = int(stats[6]['value'] or 0)
            metrics['pressing_intensity'] = min(1.0, fouls / 15)
            
            # Geçiş hızı
            dangerous_attacks = int(stats[13]['value'] or 0)
            metrics['transition_speed'] = min(1.0, dangerous_attacks / 50)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating team metrics: {str(e)}")
            return dict.fromkeys(metrics.keys(), 0.0)

    def _analyze_performance_trend(self, events: List[Dict], team_name: str) -> Dict:
        """Performans trendini analiz et"""
        try:
            periods = {
                'first_15': {'events': [], 'score': 0},
                'mid_game': {'events': [], 'score': 0},
                'last_15': {'events': [], 'score': 0}
            }
            
            for event in events:
                if event['team']['name'] != team_name:
                    continue
                    
                minute = event['time']['elapsed']
                if minute <= 15:
                    period = 'first_15'
                elif minute >= 75:
                    period = 'last_15'
                else:
                    period = 'mid_game'
                
                periods[period]['events'].append(event)
                periods[period]['score'] += self._calculate_event_impact(event)
            
            return {
                'period_scores': {k: v['score'] for k, v in periods.items()},
                'momentum': self._calculate_momentum(periods)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance trend: {str(e)}")
            return {'period_scores': {}, 'momentum': 0}

    def _calculate_event_impact(self, event: Dict) -> float:
        """Olayın etkisini hesapla"""
        try:
            impact_weights = {
                'Goal': 1.0,
                'Card': -0.3,
                'subst': 0.1,
                'Var': 0.2,
                'Shot': 0.4,
                'Save': 0.5
            }
            
            return impact_weights.get(event['type'], 0.1)
            
        except Exception as e:
            logger.error(f"Error calculating event impact: {str(e)}")
            return 0.0

    def _calculate_momentum(self, periods: Dict) -> float:
        """Momentum skorunu hesapla"""
        try:
            # Son periyotlara daha fazla ağırlık ver
            weights = {'first_15': 0.2, 'mid_game': 0.3, 'last_15': 0.5}
            momentum = sum(periods[p]['score'] * weights[p] for p in periods)
            
            return min(1.0, max(0.0, momentum))
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {str(e)}")
            return 0.0

    def _calculate_performance_score(self, metrics: Dict, trend: Dict) -> float:
        """Genel performans skorunu hesapla"""
        try:
            metric_weights = {
                'offensive_efficiency': 0.25,
                'defensive_stability': 0.25,
                'possession_control': 0.2,
                'pressing_intensity': 0.15,
                'transition_speed': 0.15
            }
            
            base_score = sum(metrics[k] * metric_weights[k] for k in metric_weights)
            momentum_bonus = trend['momentum'] * 0.2
            
            return min(1.0, max(0.0, base_score + momentum_bonus))
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {str(e)}")
            return 0.0

    def _identify_team_strengths(self, metrics: Dict) -> List[str]:
        """Takımın güçlü yönlerini belirle"""
        try:
            strengths = []
            threshold = 0.7
            
            metric_descriptions = {
                'offensive_efficiency': 'Hücum etkinliği yüksek',
                'defensive_stability': 'Savunma düzeni sağlam',
                'possession_control': 'Top kontrolü iyi',
                'pressing_intensity': 'Etkili pressing yapıyor',
                'transition_speed': 'Hızlı geçiş oyunu'
            }
            
            for metric, value in metrics.items():
                if value >= threshold:
                    strengths.append(metric_descriptions[metric])
            
            return strengths
            
        except Exception as e:
            logger.error(f"Error identifying team strengths: {str(e)}")
            return []

    def _identify_improvement_areas(self, metrics: Dict) -> List[str]:
        """Geliştirilmesi gereken alanları belirle"""
        try:
            areas = []
            threshold = 0.4
            
            metric_descriptions = {
                'offensive_efficiency': 'Hücum etkinliği artırılmalı',
                'defensive_stability': 'Savunma düzeni güçlendirilmeli',
                'possession_control': 'Top kontrolü geliştirilmeli',
                'pressing_intensity': 'Pressing sistemi iyileştirilmeli',
                'transition_speed': 'Geçiş oyunu hızlandırılmalı'
            }
            
            for metric, value in metrics.items():
                if value < threshold:
                    areas.append(metric_descriptions[metric])
            
            return areas
            
        except Exception as e:
            logger.error(f"Error identifying improvement areas: {str(e)}")
            return []

    def _evaluate_player_performances(self, player_stats: Dict) -> Dict:
        """Oyuncu performanslarını değerlendir"""
        try:
            evaluations = {}
            
            for player_id, stats in player_stats.items():
                performance_rating = min(10, max(1, stats['impact_score'] * 5))
                
                evaluations[player_id] = {
                    'name': stats['name'],
                    'team': stats['team'],
                    'rating': performance_rating,
                    'key_moments': stats['key_moments'],
                    'summary': self._generate_player_summary(stats, performance_rating)
                }
            
            return evaluations
            
        except Exception as e:
            logger.error(f"Error evaluating player performances: {str(e)}")
            return {}

    def _generate_player_summary(self, stats: Dict, rating: float) -> str:
        """Oyuncu performans özeti oluştur"""
        try:
            if rating >= 8:
                return "Maçın yıldızı, üstün performans"
            elif rating >= 6:
                return "İyi bir performans sergiliyor"
            elif rating >= 4:
                return "Ortalama bir performans"
            else:
                return "Beklentilerin altında"
                
        except Exception as e:
            logger.error(f"Error generating player summary: {str(e)}")
            return "Performans değerlendirilemedi"

    def _create_empty_analysis(self) -> Dict:
        """Boş analiz sonucu oluştur"""
        return {
            'home_team': {
                'metrics': dict.fromkeys(['offensive_efficiency', 'defensive_stability', 
                                        'possession_control', 'pressing_intensity', 
                                        'transition_speed'], 0.0),
                'trend': {'period_scores': {}, 'momentum': 0},
                'performance_score': 0.0,
                'strengths': [],
                'areas_to_improve': []
            },
            'away_team': {
                'metrics': dict.fromkeys(['offensive_efficiency', 'defensive_stability', 
                                        'possession_control', 'pressing_intensity', 
                                        'transition_speed'], 0.0),
                'trend': {'period_scores': {}, 'momentum': 0},
                'performance_score': 0.0,
                'strengths': [],
                'areas_to_improve': []
            }
        }
