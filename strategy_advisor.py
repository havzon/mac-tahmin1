import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class StrategyAdvisor:
    def __init__(self, historical_data: pd.DataFrame):
        self.data = historical_data

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