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
        metrics['attacking_strength'] = (home_goals + away_goals) / 2
        
        # Savunma istikrarı analizi
        home_conceded = team_matches[team_matches['HomeTeam'] == team]['FTAG'].mean()
        away_conceded = team_matches[team_matches['AwayTeam'] == team]['FTHG'].mean()
        metrics['defensive_stability'] = 1 - ((home_conceded + away_conceded) / 4)  # Normalize edilmiş
        
        # Diğer metrikleri hesapla (örnek veriler)
        metrics['possession_tendency'] = np.random.uniform(0.4, 0.6)  # Gerçek veri olmadığı için
        metrics['counter_attack'] = np.random.uniform(0.3, 0.7)
        metrics['set_piece_efficiency'] = np.random.uniform(0.2, 0.5)
        
        return metrics

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
