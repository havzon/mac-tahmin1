import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeamAnalyzer:
    def __init__(self):
        """Takım form ve taktik analiz sistemi"""
        logger.info("Initializing TeamAnalyzer")
        self.initialized = True
        self.team_cache = {}  # Takım verilerini önbellekleme

    def analyze_team_form(self, team_id: int, last_matches: List[Dict]) -> Dict:
        """Son maçlardaki form analizi"""
        try:
            if not last_matches:
                return self._create_empty_form_analysis()

            # Son 5 maç analizi
            recent_form = {
                'goals_scored': [],
                'goals_conceded': [],
                'shots_on_target': [],
                'possession': [],
                'home_matches': 0,
                'away_matches': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'cards': {'yellow': 0, 'red': 0},
                'xG': []  # Expected Goals
            }

            # Her maç için detaylı analiz
            for match in last_matches[-5:]:
                is_home = match['teams']['home']['id'] == team_id
                team_stats = match['statistics'][0 if is_home else 1]
                opponent_stats = match['statistics'][1 if is_home else 0]

                # Maç sonucu analizi
                if is_home:
                    recent_form['home_matches'] += 1
                    if match['goals']['home'] > match['goals']['away']:
                        recent_form['wins'] += 1
                    elif match['goals']['home'] == match['goals']['away']:
                        recent_form['draws'] += 1
                    else:
                        recent_form['losses'] += 1
                else:
                    recent_form['away_matches'] += 1
                    if match['goals']['away'] > match['goals']['home']:
                        recent_form['wins'] += 1
                    elif match['goals']['away'] == match['goals']['home']:
                        recent_form['draws'] += 1
                    else:
                        recent_form['losses'] += 1

                # İstatistik toplama
                recent_form['goals_scored'].append(match['goals']['home' if is_home else 'away'])
                recent_form['goals_conceded'].append(match['goals']['away' if is_home else 'home'])
                recent_form['shots_on_target'].append(float(team_stats['statistics'][0]['value'] or 0))
                recent_form['possession'].append(float(team_stats['statistics'][9]['value'].strip('%')) if team_stats['statistics'][9]['value'] else 50)

                # Kart analizi
                for event in match['events']:
                    if event['team']['id'] == team_id and event['type'] == 'Card':
                        if event['detail'] == 'Yellow Card':
                            recent_form['cards']['yellow'] += 1
                        elif event['detail'] == 'Red Card':
                            recent_form['cards']['red'] += 1

                # xG hesaplama
                xg = self._calculate_xG(team_stats, opponent_stats)
                recent_form['xG'].append(xg)

            # Form metriklerini hesapla
            form_metrics = {
                'overall_form': self._calculate_form_score(recent_form),
                'attack_strength': np.mean(recent_form['goals_scored']),
                'defense_stability': 1 / (np.mean(recent_form['goals_conceded']) + 0.1),
                'shooting_accuracy': np.mean(recent_form['shots_on_target']),
                'possession_control': np.mean(recent_form['possession']) / 100,
                'discipline_score': self._calculate_discipline_score(recent_form['cards']),
                'xG_performance': np.mean(recent_form['xG']),
                'home_away_ratio': recent_form['home_matches'] / max(recent_form['away_matches'], 1)
            }

            return {
                'recent_matches': recent_form,
                'metrics': form_metrics,
                'form_rating': self._calculate_overall_rating(form_metrics)
            }

        except Exception as e:
            logger.error(f"Error analyzing team form: {str(e)}")
            return self._create_empty_form_analysis()

    def analyze_tactical_patterns(self, team_id: int, last_matches: List[Dict]) -> Dict:
        """Taktik patern analizi"""
        try:
            if not last_matches:
                return self._create_empty_tactical_analysis()

            tactical_patterns = {
                'possession_style': [],
                'pressing_intensity': [],
                'attack_patterns': {
                    'wing_play': 0,
                    'central_play': 0,
                    'counter_attacks': 0
                },
                'defensive_style': {
                    'high_press': 0,
                    'mid_block': 0,
                    'low_block': 0
                },
                'formation_flexibility': 0.0
            }

            # Her maç için taktik analizi
            for match in last_matches[-5:]:
                is_home = match['teams']['home']['id'] == team_id
                team_stats = match['statistics'][0 if is_home else 1]

                # Taktik stil analizi
                possession = float(team_stats['statistics'][9]['value'].strip('%')) if team_stats['statistics'][9]['value'] else 50
                tactical_patterns['possession_style'].append(
                    'possession' if possession > 55 else 'direct' if possession < 45 else 'balanced'
                )

                # Pressing yoğunluğu
                pressing = self._calculate_pressing_intensity(team_stats)
                tactical_patterns['pressing_intensity'].append(pressing)

                # Atak paternleri
                attack_zone = self._analyze_attack_zones(match['events'], team_id)
                tactical_patterns['attack_patterns'][attack_zone] += 1

                # Savunma stili
                defense_style = self._analyze_defensive_style(team_stats)
                tactical_patterns['defensive_style'][defense_style] += 1

                # Formasyon esnekliği
                tactical_patterns['formation_flexibility'] += self._analyze_formation_changes(match['events'], team_id)

            # Taktik özeti
            dominant_style = self._calculate_dominant_style(tactical_patterns)
            tactical_rating = self._calculate_tactical_rating(tactical_patterns)

            return {
                'patterns': tactical_patterns,
                'dominant_style': dominant_style,
                'tactical_rating': tactical_rating,
                'recommendations': self._generate_tactical_recommendations(tactical_patterns)
            }

        except Exception as e:
            logger.error(f"Error analyzing tactical patterns: {str(e)}")
            return self._create_empty_tactical_analysis()

    def _calculate_xG(self, team_stats: Dict, opponent_stats: Dict) -> float:
        """Expected Goals hesaplama"""
        try:
            shots = float(team_stats['statistics'][2]['value'] or 0)
            shots_on_target = float(team_stats['statistics'][0]['value'] or 0)
            dangerous_attacks = float(team_stats['statistics'][13]['value'] or 0)

            # xG hesaplama formülü
            xg = (
                shots_on_target * 0.3 +  # İsabetli şutlar
                (shots - shots_on_target) * 0.1 +  # İsabetsiz şutlar
                dangerous_attacks * 0.05  # Tehlikeli ataklar
            )

            return min(5.0, max(0.0, xg))  # 0-5 arasında normalize et

        except Exception as e:
            logger.error(f"Error calculating xG: {str(e)}")
            return 0.0

    def _calculate_form_score(self, recent_form: Dict) -> float:
        """Form skoru hesaplama"""
        try:
            # Maç sonuçları ağırlıkları
            win_weight = 3
            draw_weight = 1
            loss_weight = 0

            # Toplam puan
            total_points = (
                recent_form['wins'] * win_weight +
                recent_form['draws'] * draw_weight +
                recent_form['losses'] * loss_weight
            )

            # Maksimum mümkün puan
            max_points = len(recent_form['goals_scored']) * win_weight

            return total_points / max(max_points, 1)

        except Exception as e:
            logger.error(f"Error calculating form score: {str(e)}")
            return 0.0

    def _calculate_discipline_score(self, cards: Dict) -> float:
        """Disiplin skoru hesaplama"""
        try:
            # Kart ağırlıkları
            yellow_weight = -0.1
            red_weight = -0.3

            # Disiplin skoru hesaplama
            discipline_score = 1.0 + (
                cards['yellow'] * yellow_weight +
                cards['red'] * red_weight
            )

            return max(0.0, min(1.0, discipline_score))

        except Exception as e:
            logger.error(f"Error calculating discipline score: {str(e)}")
            return 0.5

    def _calculate_overall_rating(self, metrics: Dict) -> float:
        """Genel performans puanı hesaplama"""
        try:
            weights = {
                'overall_form': 0.3,
                'attack_strength': 0.15,
                'defense_stability': 0.15,
                'shooting_accuracy': 0.1,
                'possession_control': 0.1,
                'discipline_score': 0.1,
                'xG_performance': 0.1
            }

            rating = sum(metrics[key] * weights[key] for key in weights)
            return min(10.0, max(0.0, rating * 10))  # 0-10 arası normalize et

        except Exception as e:
            logger.error(f"Error calculating overall rating: {str(e)}")
            return 5.0

    def _calculate_pressing_intensity(self, team_stats: Dict) -> str:
        """Pressing yoğunluğunu hesapla"""
        try:
            fouls = float(team_stats['statistics'][7]['value'] or 0)
            tackles = float(team_stats['statistics'][7]['value'] or 0)
            intensity = (fouls + tackles) / 20  # Normalize

            if intensity > 0.7:
                return 'high'
            elif intensity > 0.4:
                return 'medium'
            else:
                return 'low'

        except Exception as e:
            logger.error(f"Error calculating pressing intensity: {str(e)}")
            return 'medium'

    def _analyze_attack_zones(self, events: List[Dict], team_id: int) -> str:
        """Atak bölgelerini analiz et"""
        try:
            wing_attacks = 0
            central_attacks = 0
            counter_attacks = 0

            for event in events:
                if event['team']['id'] == team_id and event['type'] in ['Goal', 'Shot']:
                    # Pozisyon bazlı analiz
                    if 'position' in event:
                        if event['position']['x'] < 30:  # Kanat
                            wing_attacks += 1
                        else:  # Merkez
                            central_attacks += 1

                    # Hızlı atak analizi
                    if 'time_between_events' in event and event['time_between_events'] < 10:
                        counter_attacks += 1

            # En baskın atak stilini belirle
            max_attacks = max(wing_attacks, central_attacks, counter_attacks)
            if max_attacks == wing_attacks:
                return 'wing_play'
            elif max_attacks == counter_attacks:
                return 'counter_attacks'
            else:
                return 'central_play'

        except Exception as e:
            logger.error(f"Error analyzing attack zones: {str(e)}")
            return 'central_play'

    def _analyze_defensive_style(self, team_stats: Dict) -> str:
        """Savunma stilini analiz et"""
        try:
            possession = float(team_stats['statistics'][9]['value'].strip('%')) if team_stats['statistics'][9]['value'] else 50
            tackles = float(team_stats['statistics'][7]['value'] or 0)
            interceptions = float(team_stats['statistics'][8]['value'] or 0)

            # Savunma yüksekliğini belirle
            defensive_height = (
                tackles * 0.4 +
                interceptions * 0.3 +
                (100 - possession) * 0.3
            ) / 100

            if defensive_height > 0.7:
                return 'high_press'
            elif defensive_height > 0.4:
                return 'mid_block'
            else:
                return 'low_block'

        except Exception as e:
            logger.error(f"Error analyzing defensive style: {str(e)}")
            return 'mid_block'

    def _analyze_formation_changes(self, events: List[Dict], team_id: int) -> float:
        """Formasyon değişikliklerini analiz et"""
        try:
            formation_changes = 0
            substitutions = 0

            for event in events:
                if event['team']['id'] == team_id:
                    if event['type'] == 'subst':
                        substitutions += 1
                    elif 'formation_change' in event:
                        formation_changes += 1

            # Esneklik skoru hesaplama
            flexibility = (formation_changes * 0.7 + substitutions * 0.3) / 5
            return min(1.0, flexibility)

        except Exception as e:
            logger.error(f"Error analyzing formation changes: {str(e)}")
            return 0.0

    def _calculate_dominant_style(self, patterns: Dict) -> Dict:
        """Baskın oyun stilini belirle"""
        try:
            # Possession stili analizi
            possession_styles = patterns['possession_style']
            dominant_possession = max(set(possession_styles), key=possession_styles.count)

            # Pressing analizi
            pressing_styles = patterns['pressing_intensity']
            dominant_pressing = max(set(pressing_styles), key=pressing_styles.count)

            # Atak stili
            attack_patterns = patterns['attack_patterns']
            dominant_attack = max(attack_patterns.items(), key=lambda x: x[1])[0]

            # Savunma stili
            defensive_patterns = patterns['defensive_style']
            dominant_defense = max(defensive_patterns.items(), key=lambda x: x[1])[0]

            return {
                'possession_style': dominant_possession,
                'pressing_style': dominant_pressing,
                'attack_style': dominant_attack,
                'defensive_style': dominant_defense,
                'formation_flexibility': patterns['formation_flexibility']
            }

        except Exception as e:
            logger.error(f"Error calculating dominant style: {str(e)}")
            return {
                'possession_style': 'balanced',
                'pressing_style': 'medium',
                'attack_style': 'balanced',
                'defensive_style': 'mid_block',
                'formation_flexibility': 0.5
            }

    def _calculate_tactical_rating(self, patterns: Dict) -> float:
        """Taktik rating hesaplama"""
        try:
            # Taktik tutarlılığı
            style_consistency = len(set(patterns['possession_style'])) / len(patterns['possession_style'])
            pressing_consistency = len(set(patterns['pressing_intensity'])) / len(patterns['pressing_intensity'])

            # Çeşitlilik skoru
            attack_variety = sum(1 for v in patterns['attack_patterns'].values() if v > 0) / len(patterns['attack_patterns'])
            defense_variety = sum(1 for v in patterns['defensive_style'].values() if v > 0) / len(patterns['defensive_style'])

            # Esneklik
            flexibility = patterns['formation_flexibility']

            # Toplam taktik puanı
            tactical_score = (
                style_consistency * 0.2 +
                pressing_consistency * 0.2 +
                attack_variety * 0.2 +
                defense_variety * 0.2 +
                flexibility * 0.2
            )

            return min(10.0, tactical_score * 10)  # 0-10 arası normalize et

        except Exception as e:
            logger.error(f"Error calculating tactical rating: {str(e)}")
            return 5.0

    def _generate_tactical_recommendations(self, patterns: Dict) -> List[str]:
        """Taktik önerileri oluştur"""
        try:
            recommendations = []

            # Possession bazlı öneriler
            possession_styles = patterns['possession_style']
            if possession_styles.count('possession') > len(possession_styles) * 0.6:
                recommendations.append("Top kontrolü yüksek, ancak daha etkili hücum geçişleri gerekebilir")
            elif possession_styles.count('direct') > len(possession_styles) * 0.6:
                recommendations.append("Direkt oyun stili, top kontrolünü artırmak faydalı olabilir")

            # Pressing bazlı öneriler
            pressing_intensity = patterns['pressing_intensity']
            if pressing_intensity.count('high') > len(pressing_intensity) * 0.7:
                recommendations.append("Yüksek pressing enerji tüketimini artırıyor, seçici pressing düşünülebilir")
            elif pressing_intensity.count('low') > len(pressing_intensity) * 0.7:
                recommendations.append("Daha agresif pressing rakibi zorlayabilir")

            # Atak çeşitliliği önerileri
            attack_patterns = patterns['attack_patterns']
            if max(attack_patterns.values()) > sum(attack_patterns.values()) * 0.7:
                recommendations.append("Hücum çeşitliliğini artırmak savunmayı zorlamada faydalı olabilir")

            # Defansif öneriler
            defensive_style = patterns['defensive_style']
            if defensive_style['high_press'] > sum(defensive_style.values()) * 0.7:
                recommendations.append("Yüksek savunma hattı risk oluşturabilir, duruma göre orta blok düşünülebilir")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating tactical recommendations: {str(e)}")
            return ["Taktik önerileri oluşturulamadı"]

    def _create_empty_form_analysis(self) -> Dict:
        """Boş form analizi oluştur"""
        return {
            'recent_matches': {
                'goals_scored': [],
                'goals_conceded': [],
                'shots_on_target': [],
                'possession': [],
                'home_matches': 0,
                'away_matches': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'cards': {'yellow': 0, 'red': 0},
                'xG': []
            },
            'metrics': {
                'overall_form': 0.0,
                'attack_strength': 0.0,
                'defense_stability': 0.0,
                'shooting_accuracy': 0.0,
                'possession_control': 0.0,
                'discipline_score': 0.0,
                'xG_performance': 0.0,
                'home_away_ratio': 1.0
            },
            'form_rating': 5.0
        }

    def _create_empty_tactical_analysis(self) -> Dict:
        """Boş taktik analizi oluştur"""
        return {
            'patterns': {
                'possession_style': [],
                'pressing_intensity': [],
                'attack_patterns': {
                    'wing_play': 0,
                    'central_play': 0,
                    'counter_attacks': 0
                },
                'defensive_style': {
                    'high_press': 0,
                    'mid_block': 0,
                    'low_block': 0
                },
                'formation_flexibility': 0.0
            },
            'dominant_style': {
                'possession_style': 'balanced',
                'pressing_style': 'medium',
                'attack_style': 'balanced',
                'defensive_style': 'mid_block',
                'formation_flexibility': 0.5
            },
            'tactical_rating': 5.0,
            'recommendations': ["Yeterli veri yok"]
        }
