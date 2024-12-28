import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)


class StrategyAdvisor:
    def __init__(self, historical_data: pd.DataFrame):
        self.data = historical_data

    def get_team_form(self, data: pd.DataFrame, team: str) -> Dict:
        """Takımın son form durumunu analiz et"""
        try:
            # Son 5 maçı al
            team_matches = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)].tail(5)

            if len(team_matches) == 0:
                return self._create_empty_form()

            form_data = {
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'goals_scored': [],
                'goals_conceded': [],
                'points': [],
                'form_trend': [],
                'last_matches': []
            }

            # Son maçları analiz et
            for _, match in team_matches.iterrows():
                is_home = match['HomeTeam'] == team
                team_goals = match['FTHG'] if is_home else match['FTAG']
                opponent_goals = match['FTAG'] if is_home else match['FTHG']

                # Gol istatistikleri
                form_data['goals_scored'].append(float(team_goals))
                form_data['goals_conceded'].append(float(opponent_goals))

                # Maç sonucu
                if team_goals > opponent_goals:
                    form_data['wins'] += 1
                    form_data['points'].append(3)
                    form_data['form_trend'].append(1)  # Galibiyet
                elif team_goals == opponent_goals:
                    form_data['draws'] += 1
                    form_data['points'].append(1)
                    form_data['form_trend'].append(0)  # Beraberlik
                else:
                    form_data['losses'] += 1
                    form_data['points'].append(0)
                    form_data['form_trend'].append(-1)  # Mağlubiyet

                # Maç detayı
                form_data['last_matches'].append({
                    'opponent': match['AwayTeam'] if is_home else match['HomeTeam'],
                    'score': f"{team_goals}-{opponent_goals}",
                    'home_away': 'H' if is_home else 'A'
                })

            # Form metrikleri hesapla
            form_data.update({
                'avg_goals_scored': np.mean(form_data['goals_scored']),
                'avg_goals_conceded': np.mean(form_data['goals_conceded']),
                'total_points': sum(form_data['points']),
                'form_score': self._calculate_form_score(form_data['form_trend']),
                'current_streak': self._calculate_streak(form_data['form_trend'])
            })

            return form_data

        except Exception as e:
            logger.error(f"Error calculating team form: {str(e)}")
            return self._create_empty_form()

    def _create_empty_form(self) -> Dict:
        """Boş form verisi oluştur"""
        return {
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_scored': [],
            'goals_conceded': [],
            'points': [],
            'form_trend': [],
            'last_matches': [],
            'avg_goals_scored': 0,
            'avg_goals_conceded': 0,
            'total_points': 0,
            'form_score': 0,
            'current_streak': 'N/A'
        }

    def _calculate_form_score(self, trends: List[int]) -> float:
        """Form skorunu hesapla (0-1 arası)"""
        if not trends:
            return 0.0

        # Son maçlara daha fazla ağırlık ver
        weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])[:len(trends)]
        weights = weights / weights.sum()  # Normalize

        # Trend değerlerini 0-1 aralığına dönüştür
        normalized_trends = [(x + 1) / 2 for x in trends]  # -1->0, 0->0.5, 1->1

        return float(np.sum(weights * normalized_trends))

    def _calculate_streak(self, trends: List[int]) -> str:
        """Mevcut galibiyet/mağlubiyet serisini hesapla"""
        if not trends:
            return 'N/A'

        current = trends[-1]
        streak = 1

        for i in range(len(trends)-2, -1, -1):
            if trends[i] == current:
                streak += 1
            else:
                break

        streak_type = 'G' if current == 1 else 'M' if current == -1 else 'B'
        return f"{streak}{streak_type}"

    def analyze_team_style(self, team: str) -> Dict[str, float]:
        """Takımın oyun stilini analiz et"""
        team_matches = self.data[(self.data['HomeTeam'] == team) | (self.data['AwayTeam'] == team)].tail(10)

        # Oyun stili metrikleri
        metrics = {
            'attacking_strength': 0.0,  # Hücum gücü
            'defensive_stability': 0.0,  # Savunma istikrarı
            'possession_tendency': 0.0,  # Top kontrolü eğilimi
            'counter_attack': 0.0,      # Kontra atak etkinliği
            'set_piece_efficiency': 0.0  # Duran top etkinliği
        }

        if len(team_matches) == 0:
            return metrics

        # Hücum gücü analizi
        home_goals = team_matches[team_matches['HomeTeam'] == team]['FTHG'].mean()
        away_goals = team_matches[team_matches['AwayTeam'] == team]['FTAG'].mean()
        metrics['attacking_strength'] = min(1.0, max(0.0, (home_goals + away_goals) / 4))

        # Savunma istikrarı analizi
        home_conceded = team_matches[team_matches['HomeTeam'] == team]['FTAG'].mean()
        away_conceded = team_matches[team_matches['AwayTeam'] == team]['FTHG'].mean()
        metrics['defensive_stability'] = min(1.0, max(0.0, 1 - ((home_conceded + away_conceded) / 4)))

        # Diğer metrikleri hesapla
        recent_matches = team_matches.tail(5)
        metrics['possession_tendency'] = self._calculate_possession_tendency(recent_matches, team)
        metrics['counter_attack'] = self._calculate_counter_attack_efficiency(recent_matches, team)
        metrics['set_piece_efficiency'] = self._calculate_set_piece_efficiency(recent_matches, team)

        return metrics

    def analyze_prediction_reliability(self, home_team: str, away_team: str, prediction_probs: List[float]) -> Dict:
        """Tahmin güvenilirliğini kural tabanlı sistem ile analiz et"""
        # Son karşılaşmaları al
        h2h_matches = self.data[
            ((self.data['HomeTeam'] == home_team) & (self.data['AwayTeam'] == away_team)) |
            ((self.data['HomeTeam'] == away_team) & (self.data['AwayTeam'] == home_team))
        ].tail(5)

        # Takım formlarını al
        home_form = self.analyze_team_style(home_team)
        away_form = self.analyze_team_style(away_team)

        # Güven faktörlerini hesapla
        confidence_factors = []
        risk_factors = []
        reliability_score = 0.5  # Başlangıç skoru

        # Form analizi
        if home_form['attacking_strength'] > 0.6:
            confidence_factors.append(f"{home_team} son maçlarda yüksek gol performansı gösteriyor")
            reliability_score += 0.1
        if away_form['defensive_stability'] < 0.4:
            confidence_factors.append(f"{away_team} savunmada zorluk yaşıyor")
            reliability_score += 0.1

        # Head-to-head analizi
        if len(h2h_matches) >= 3:
            confidence_factors.append("Yeterli head-to-head veri mevcut")
            reliability_score += 0.1

            # Son maçlardaki skorları analiz et
            home_wins = len(h2h_matches[h2h_matches['FTR'] == 'H'])
            away_wins = len(h2h_matches[h2h_matches['FTR'] == 'A'])
            if home_wins > away_wins and prediction_probs[0] > prediction_probs[2]:
                confidence_factors.append(f"{home_team} head-to-head geçmişte üstün")
                reliability_score += 0.1
            elif away_wins > home_wins and prediction_probs[2] > prediction_probs[0]:
                confidence_factors.append(f"{away_team} head-to-head geçmişte üstün")
                reliability_score += 0.1
        else:
            risk_factors.append("Yetersiz head-to-head veri")
            reliability_score -= 0.1

        # Tahmin olasılıkları analizi
        max_prob = max(prediction_probs)
        if max_prob > 0.6:
            confidence_factors.append("Belirgin bir favori var")
            reliability_score += 0.1
        elif max(prediction_probs) < 0.4:
            risk_factors.append("Sonuç belirsizliği yüksek")
            reliability_score -= 0.1

        # Form tutarlılığı kontrolü
        if self._check_form_consistency(home_team) and self._check_form_consistency(away_team):
            confidence_factors.append("Her iki takım da tutarlı form gösteriyor")
            reliability_score += 0.1
        else:
            risk_factors.append("Takımların form grafiği değişken")
            reliability_score -= 0.1

        # Skor normalizasyonu
        reliability_score = min(1.0, max(0.0, reliability_score))

        # Tavsiye oluştur
        recommendation = self._generate_recommendation(reliability_score, confidence_factors, risk_factors)

        return {
            "reliability_score": reliability_score,
            "confidence_factors": confidence_factors,
            "risk_factors": risk_factors,
            "recommendation": recommendation
        }

    def _check_form_consistency(self, team: str) -> bool:
        """Takımın form tutarlılığını kontrol et"""
        recent_matches = self.data[(self.data['HomeTeam'] == team) | (self.data['AwayTeam'] == team)].tail(5)
        if len(recent_matches) < 3:
            return False

        # Form puanlarını hesapla
        form_points = []
        for _, match in recent_matches.iterrows():
            if match['HomeTeam'] == team:
                form_points.append(3 if match['FTR'] == 'H' else 1 if match['FTR'] == 'D' else 0)
            else:
                form_points.append(3 if match['FTR'] == 'A' else 1 if match['FTR'] == 'D' else 0)

        # Standart sapmayı kontrol et
        return np.std(form_points) < 1.5

    def _generate_recommendation(self, reliability_score: float, confidence_factors: List[str], risk_factors: List[str]) -> str:
        """Analiz sonuçlarına göre tavsiye oluştur"""
        if reliability_score > 0.7:
            return "Yüksek güvenilirlikli tahmin, belirlenen oranlarla bahis düşünülebilir"
        elif reliability_score > 0.5:
            return "Orta güvenilirlikli tahmin, düşük oranlarla bahis düşünülebilir"
        else:
            return "Düşük güvenilirlikli tahmin, bahis önerilmez"

    def _calculate_possession_tendency(self, matches: pd.DataFrame, team: str) -> float:
        """Top kontrolü eğilimini hesapla"""
        # Basit bir örnek - gerçek veri olmadığı için
        return np.random.uniform(0.4, 0.6)

    def _calculate_counter_attack_efficiency(self, matches: pd.DataFrame, team: str) -> float:
        """Kontra atak etkinliğini hesapla"""
        return np.random.uniform(0.3, 0.7)

    def _calculate_set_piece_efficiency(self, matches: pd.DataFrame, team: str) -> float:
        """Duran top etkinliğini hesapla"""
        return np.random.uniform(0.2, 0.5)

    def generate_tactical_advice(self, home_team: str, away_team: str) -> Dict[str, List[str]]:
        """Takımlar için taktik önerileri oluştur"""
        home_style = self.analyze_team_style(home_team)
        away_style = self.analyze_team_style(away_team)

        advice = {
            'attacking': [],
            'defensive': [],
            'general': []
        }

        # Hücum önerileri
        if home_style['attacking_strength'] > away_style['defensive_stability']:
            advice['attacking'].append("Rakibin savunma zaafiyetlerini değerlendirmek için yüksek tempo ile başlayın")
            advice['attacking'].append("Kanat ataklarına ağırlık verin")
        else:
            advice['attacking'].append("Kontrollü hücum organizasyonları kurun")
            advice['attacking'].append("Kontra atak fırsatlarını değerlendirin")

        # Savunma önerileri
        if away_style['attacking_strength'] > home_style['defensive_stability']:
            advice['defensive'].append("Savunmada ekstra önlem alın")
            advice['defensive'].append("Defansif orta saha desteğini artırın")
        else:
            advice['defensive'].append("Normal savunma düzeninizi koruyun")
            advice['defensive'].append("Rakibin kontra ataklarına dikkat edin")

        # Genel öneriler
        if home_style['possession_tendency'] > 0.5:
            advice['general'].append("Top kontrolünü elinizde tutmaya çalışın")
        else:
            advice['general'].append("Hızlı geçiş oyununa odaklanın")

        return advice

    def get_key_player_roles(self, team: str) -> List[str]:
        """Takım için önemli oyuncu rollerini belirle"""
        team_style = self.analyze_team_style(team)

        key_roles = []
        if team_style['attacking_strength'] > 0.6:
            key_roles.append("Hızlı kanat oyuncuları")
        if team_style['defensive_stability'] < 0.4:
            key_roles.append("Defansif orta saha")
        if team_style['set_piece_efficiency'] > 0.5:
            key_roles.append("Duran top uzmanı")

        return key_roles

    def predict_match_tempo(self, home_team: str, away_team: str) -> str:
        """Maçın tempo tahminini yap"""
        home_style = self.analyze_team_style(home_team)
        away_style = self.analyze_team_style(away_team)

        combined_attack = (home_style['attacking_strength'] + away_style['attacking_strength']) / 2
        combined_defense = (home_style['defensive_stability'] + away_style['defensive_stability']) / 2

        if combined_attack > 0.6 and combined_defense < 0.4:
            return "Yüksek tempolu, gol pozisyonu bol bir maç bekleniyor"
        elif combined_attack < 0.3 and combined_defense > 0.7:
            return "Düşük tempolu, az gollü bir maç bekleniyor"
        else:
            return "Orta tempolu, dengeli bir maç bekleniyor"