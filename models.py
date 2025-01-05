import pandas as pd
import numpy as np
import math
from typing import List, Dict, Tuple
import logging
import scipy.stats as stats

class StatisticalModel:
    def __init__(self):
        """İstatistiksel tahmin modeli"""
        pass

    def calculate_probabilities(self, df: pd.DataFrame, home_team: str, away_team: str) -> np.ndarray:
        """Maç sonucu olasılıklarını hesapla"""
        try:
            # Takım istatistiklerini al
        home_stats = self._get_team_stats(df, home_team)
        away_stats = self._get_team_stats(df, away_team)

            # Son 10 maçtaki galibiyet/mağlubiyet oranlarını hesapla
            home_matches = df[(df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)].tail(10)
            away_matches = df[(df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)].tail(10)
            
            # Ev sahibi takımın performansı
            home_wins = 0
            home_losses = 0
            for _, match in home_matches.iterrows():
                if match['HomeTeam'] == home_team:
                    if match['FTR'] == 'H':
                        home_wins += 1
                    elif match['FTR'] == 'A':
                        home_losses += 1
                else:
                    if match['FTR'] == 'A':
                        home_wins += 1
                    elif match['FTR'] == 'H':
                        home_losses += 1
            
            # Deplasman takımının performansı
            away_wins = 0
            away_losses = 0
            for _, match in away_matches.iterrows():
                if match['HomeTeam'] == away_team:
                    if match['FTR'] == 'H':
                        away_wins += 1
                    elif match['FTR'] == 'A':
                        away_losses += 1
                else:
                    if match['FTR'] == 'A':
                        away_wins += 1
                    elif match['FTR'] == 'H':
                        away_losses += 1

            # Galibiyet/mağlubiyet oranları
            home_win_rate = home_wins / len(home_matches)
            home_loss_rate = home_losses / len(home_matches)
            away_win_rate = away_wins / len(away_matches)
            away_loss_rate = away_losses / len(away_matches)

            # Temel olasılıkları hesapla
            p_home = (home_win_rate * 0.3 + 
                     home_stats['recent_form'] * 0.3 + 
                     away_loss_rate * 0.2 +
                     (home_stats['avg_goals_scored'] / (away_stats['avg_goals_scored'] + 0.1)) * 0.2)

            p_away = (away_win_rate * 0.3 + 
                     away_stats['recent_form'] * 0.3 + 
                     home_loss_rate * 0.2 +
                     (away_stats['avg_goals_scored'] / (home_stats['avg_goals_scored'] + 0.1)) * 0.2)

            # Oran bazlı düzeltme
            if home_stats['avg_odds'] > 0 and away_stats['avg_odds'] > 0:
                odds_ratio = home_stats['avg_odds'] / away_stats['avg_odds']
                odds_adjustment = 0.1 * (1 - odds_ratio)  # -0.1 ile +0.1 arası
                p_home += odds_adjustment
                p_away -= odds_adjustment

            # Gol trendi etkisi
            goal_trend_effect = (home_stats['goal_trend'] - away_stats['goal_trend']) * 0.1
            p_home += goal_trend_effect
            p_away -= goal_trend_effect

            # Beraberlik olasılığı
        p_draw = 1 - (p_home + p_away)

            # Olasılıkları normalize et
        total = p_home + p_draw + p_away
            p_home = max(0.05, min(0.90, p_home / total))
            p_away = max(0.05, min(0.90, p_away / total))
            p_draw = 1 - (p_home + p_away)

            return np.array([p_home, p_draw, p_away])

        except Exception as e:
            logging.error(f"Olasılık hesaplama hatası: {str(e)}")
            raise e

    def _get_team_stats(self, df: pd.DataFrame, team: str) -> Dict[str, float]:
        """Takım istatistiklerini hesapla"""
        try:
        # Son 10 maçı al
        recent_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(10)

            if len(recent_matches) == 0:
                raise ValueError(f"{team} için maç verisi bulunamadı")

            # Detaylı istatistikler
            stats = {
                'goals_scored': [],
                'goals_conceded': [],
                'first_half_goals': [],
                'first_half_conceded': [],
                'corners_for': [],
                'corners_against': [],
                'cards': [],
                'odds': []
            }
            
            # Son 10 maçın detaylı analizi
            for _, match in recent_matches.iterrows():
                if match['HomeTeam'] == team:
                    stats['goals_scored'].append(float(match['FTHG']))
                    stats['goals_conceded'].append(float(match['FTAG']))
                    stats['first_half_goals'].append(float(match['HTHG']))
                    stats['first_half_conceded'].append(float(match['HTAG']))
                    stats['odds'].append(float(match['AvgH']))
                else:
                    stats['goals_scored'].append(float(match['FTAG']))
                    stats['goals_conceded'].append(float(match['FTHG']))
                    stats['first_half_goals'].append(float(match['HTAG']))
                    stats['first_half_conceded'].append(float(match['HTHG']))
                    stats['odds'].append(float(match['AvgA']))

            # Ağırlıklı ortalamalar (son maçlara daha fazla ağırlık ver)
            weights = np.linspace(0.5, 1.0, len(stats['goals_scored']))
            
            # İstatistikleri hesapla
            avg_stats = {
                'avg_goals_scored': np.average(stats['goals_scored'], weights=weights),
                'avg_goals_conceded': np.average(stats['goals_conceded'], weights=weights),
                'avg_first_half_goals': np.average(stats['first_half_goals'], weights=weights),
                'avg_first_half_conceded': np.average(stats['first_half_conceded'], weights=weights),
                'avg_odds': np.mean(stats['odds']),  # Ortalama oran
                'form_goals': np.mean(stats['goals_scored'][-3:]),  # Son 3 maçtaki gol ortalaması
                'form_conceded': np.mean(stats['goals_conceded'][-3:]),  # Son 3 maçta yenen gol
                'goal_trend': np.mean(stats['goals_scored'][-3:]) - np.mean(stats['goals_scored'][:3]),  # Gol trendi
                'defense_trend': np.mean(stats['goals_conceded'][-3:]) - np.mean(stats['goals_conceded'][:3])  # Savunma trendi
            }
            
            # Form puanı hesapla
        form_points = []
            for i in range(min(5, len(recent_matches))):
                match = recent_matches.iloc[i]
            if match['HomeTeam'] == team:
                form_points.append(3 if match['FTR'] == 'H' else 1 if match['FTR'] == 'D' else 0)
            else:
                form_points.append(3 if match['FTR'] == 'A' else 1 if match['FTR'] == 'D' else 0)

            avg_stats['recent_form'] = sum(form_points) / (len(form_points) * 3)
            
            return avg_stats
            
        except Exception as e:
            logging.error(f"Takım istatistikleri hesaplanırken hata: {str(e)}")
            raise e

    def predict_goals(self, df, home_team, away_team):
        try:
            # Son 20 maçı al
            home_matches = df[(df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)].tail(20)
            away_matches = df[(df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)].tail(20)
            
            if len(home_matches) == 0 or len(away_matches) == 0:
                raise ValueError("Yeterli maç verisi bulunamadı")

            # Gol analizleri
            home_goals_scored = []
            home_goals_conceded = []
            away_goals_scored = []
            away_goals_conceded = []
            
            # Ev sahibi analizi
            for _, match in home_matches.iterrows():
                if match['HomeTeam'] == home_team:
                    home_goals_scored.append(float(match['FTHG']))
                    home_goals_conceded.append(float(match['FTAG']))
                else:
                    home_goals_scored.append(float(match['FTAG']))
                    home_goals_conceded.append(float(match['FTHG']))

            # Deplasman analizi
            for _, match in away_matches.iterrows():
                if match['HomeTeam'] == away_team:
                    away_goals_scored.append(float(match['FTHG']))
                    away_goals_conceded.append(float(match['FTAG']))
                else:
                    away_goals_scored.append(float(match['FTAG']))
                    away_goals_conceded.append(float(match['FTHG']))

            # Son 5 maç ve son 20 maç karşılaştırması (form trendi)
            home_recent_form = np.mean(home_goals_scored[:5]) / (np.mean(home_goals_scored) + 0.01)
            away_recent_form = np.mean(away_goals_scored[:5]) / (np.mean(away_goals_scored) + 0.01)

            # Gol ortalamaları (son maçlara daha fazla ağırlık ver)
            weights = np.linspace(0.5, 1.0, len(home_goals_scored))
            home_avg_scored = np.average(home_goals_scored, weights=weights)
            home_avg_conceded = np.average(home_goals_conceded, weights=weights)
            away_avg_scored = np.average(away_goals_scored, weights=weights)
            away_avg_conceded = np.average(away_goals_conceded, weights=weights)

            # Bahis oranlarından gol beklentisi
            if 'B365>2.5' in df.columns and 'B365<2.5' in df.columns:
                recent_match = home_matches.iloc[0] if not home_matches.empty else None
                if recent_match is not None:
                    over_odds = float(recent_match['B365>2.5'])
                    under_odds = float(recent_match['B365<2.5'])
                    odds_expectation = (1/over_odds) / (1/over_odds + 1/under_odds) * 2.5
                else:
                    odds_expectation = 2.5
            else:
                odds_expectation = 2.5

            # Takımların gol atma/yeme eğilimleri
            home_scoring_power = home_avg_scored * home_recent_form
            home_conceding_weakness = home_avg_conceded
            away_scoring_power = away_avg_scored * away_recent_form
            away_conceding_weakness = away_avg_conceded

            # Beklenen gol sayısı hesaplama
            expected_home_goals = (home_scoring_power * away_conceding_weakness) * 1.1  # Ev sahibi avantajı
            expected_away_goals = (away_scoring_power * home_conceding_weakness)
            expected_total = expected_home_goals + expected_away_goals

            # Form bazlı düzeltme
            form_factor = (home_recent_form + away_recent_form) / 2
            expected_total = expected_total * form_factor

            # Alt/üst olasılıkları
            over_probs = []
            for threshold in [0.5, 1.5, 2.5, 3.5]:
                # Poisson olasılığı
                prob = 1 - stats.poisson.cdf(threshold, expected_total)
                
                # Form ve skor eğilimine göre düzelt
                prob = prob * form_factor
                
                # Olasılığı sınırla
                prob = min(max(prob, 0.05), 0.95)
                over_probs.append(prob)

            return expected_total, over_probs

        except Exception as e:
            logging.error(f"Gol tahmini hesaplanırken hata: {str(e)}")
            return 2.5, [0.8, 0.6, 0.4, 0.2]  # Varsayılan değerler

    def predict_first_half_goals(self, match_stats: Dict, events: List[Dict]) -> Dict:
        """İlk yarı gol tahmini yap"""
        try:
            if not match_stats or len(match_stats) < 2:
                return self._create_default_prediction()

            home_stats = self._extract_statistics(match_stats[0]['statistics'])
            away_stats = self._extract_statistics(match_stats[1]['statistics'])

            # Temel metrikleri hesapla
            home_attack = self._calculate_attack_strength(home_stats)
            away_attack = self._calculate_attack_strength(away_stats)
            home_defense = self._calculate_defense_strength(home_stats)
            away_defense = self._calculate_defense_strength(away_stats)

            # Maç özgü faktörler
            home_advantage = 1.1  # Ev sahibi avantajı
            recent_form = self._calculate_recent_form(events)
            weather_impact = self._get_weather_impact(match_stats)

            # İlk yarı gol tahmini
            base_goals = (home_attack * away_defense + away_attack * home_defense) / 2
            predicted_goals = base_goals * home_advantage * recent_form * weather_impact
            predicted_goals = max(0, min(3, predicted_goals))  # 0-3 aralığında sınırla

            # Bahis olasılıkları
            probabilities = {
                'over_0.5': self._calculate_probability(predicted_goals, 0.5),
                'over_1.5': self._calculate_probability(predicted_goals, 1.5),
                'over_2.5': self._calculate_probability(predicted_goals, 2.5)
            }

            # Güven skoru
            confidence = self._calculate_confidence(home_stats, away_stats)

            return {
                'predicted_goals': round(predicted_goals, 2),
                'probabilities': probabilities,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error predicting first half goals: {str(e)}")
            return self._create_default_prediction()

    def _extract_statistics(self, stats: List[Dict]) -> Dict:
        """İstatistikleri daha kullanışlı bir formata dönüştür"""
        extracted = {}
        for stat in stats:
            try:
                value = stat['value']
                if '%' in value:
                    value = float(value.replace('%', '')) / 100
                else:
                    value = float(value)
                extracted[stat['type']] = value
            except (ValueError, KeyError):
                continue
        return extracted

    def _calculate_attack_strength(self, stats: Dict) -> float:
        """Hücum gücünü hesapla"""
        shots_on_goal = stats.get('Shots on Goal', 0)
        shots_off_goal = stats.get('Shots off Goal', 0)
        corners = stats.get('Corner Kicks', 0)
        return (shots_on_goal + 0.5 * shots_off_goal + 0.2 * corners) / 10

    def _calculate_defense_strength(self, stats: Dict) -> float:
        """Defans gücünü hesapla"""
        blocked_shots = stats.get('Blocked Shots', 0)
        fouls = stats.get('Fouls', 0)
        yellow_cards = stats.get('Yellow Cards', 0)
        return (blocked_shots + 0.5 * fouls + 0.2 * yellow_cards) / 10

    def _calculate_recent_form(self, events: List[Dict]) -> float:
        """Son maçlardaki performansa göre form faktörü hesapla"""
        if not events:
            return 1.0  # Varsayılan değer

        # Son 5 maçın gol ortalaması
        recent_goals = [e['goals'] for e in events[-5:]]
        avg_goals = sum(recent_goals) / len(recent_goals) if recent_goals else 1.0
        return min(1.5, max(0.5, avg_goals))  # 0.5-1.5 aralığında sınırla

    def _get_weather_impact(self, match_stats: Dict) -> float:
        """Hava durumunun gol tahminine etkisini hesapla"""
        weather = match_stats.get('weather', {}).get('main', 'Clear')
        weather_impact = {
            'Clear': 1.0,
            'Rain': 0.8,
            'Snow': 0.6,
            'Fog': 0.9,
            'Clouds': 0.95
        }
        return weather_impact.get(weather, 1.0)

    def _calculate_probability(self, predicted_goals: float, threshold: float) -> float:
        """Belirli bir gol eşiği için olasılık hesapla"""
        return min(1.0, max(0.0, 1 - np.exp(-predicted_goals * threshold)))

    def _calculate_confidence(self, home_stats: Dict, away_stats: Dict) -> float:
        """Tahminin güvenilirliğini hesapla"""
        total_shots = home_stats.get('Total Shots', 0) + away_stats.get('Total Shots', 0)
        possession_diff = abs(home_stats.get('Ball Possession', 50) - away_stats.get('Ball Possession', 50))
        return min(1.0, max(0.0, (total_shots / 30) * (1 - possession_diff / 100)))

    def _create_default_prediction(self) -> Dict:
        """Varsayılan tahmini oluştur"""
        return {
            'predicted_goals': 1.0,
            'probabilities': {
                'over_0.5': 0.5,
                'over_1.5': 0.3,
                'over_2.5': 0.1
            },
            'confidence': 0.5
        }

    def _calculate_match_tempo(self, matches, team):
        """Maç temposunu hesapla"""
        try:
            tempo_scores = []
            for _, match in matches.iterrows():
                if match['HomeTeam'] == team:
                    shots = float(match.get('HS', 10))
                    shots_target = float(match.get('HST', 4))
                    possession = float(match.get('HPOS', 50))
                else:
                    shots = float(match.get('AS', 10))
                    shots_target = float(match.get('AST', 4))
                    possession = float(match.get('APOS', 50))
                
                # Tempo skoru hesaplama (0-1 arası)
                shot_score = (shots + shots_target) / 25  # Normalize et
                possession_score = possession / 100
                tempo_score = (shot_score * 0.7 + possession_score * 0.3)
                tempo_scores.append(tempo_score)
            
            # Son maçlara daha fazla ağırlık ver
            weights = np.linspace(0.5, 1.0, len(tempo_scores))
            return np.average(tempo_scores, weights=weights)
            
        except Exception as e:
            logging.error(f"Maç temposu hesaplanırken hata: {str(e)}")
            return 0.5  # Varsayılan değer

    def predict_both_teams_to_score(self, df, home_team, away_team):
        try:
            # Son 10 maçtaki BTTS istatistiklerini al
            home_matches = df[(df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)].tail(10)
            away_matches = df[(df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)].tail(10)
            
            # BTTS oranlarını hesapla
            home_btts = 0
            for _, match in home_matches.iterrows():
                if (match['HomeTeam'] == home_team and match['FTHG'] > 0 and match['FTAG'] > 0) or \
                   (match['AwayTeam'] == home_team and match['FTHG'] > 0 and match['FTAG'] > 0):
                    home_btts += 1
            
            away_btts = 0
            for _, match in away_matches.iterrows():
                if (match['HomeTeam'] == away_team and match['FTHG'] > 0 and match['FTAG'] > 0) or \
                   (match['AwayTeam'] == away_team and match['FTHG'] > 0 and match['FTAG'] > 0):
                    away_btts += 1
            
            # Ağırlıklı BTTS olasılığı
            btts_prob = (home_btts/10 * 0.6 + away_btts/10 * 0.4)
            
            # Olasılığı 0.05-0.95 arasında tut
            return min(max(btts_prob, 0.05), 0.95)
            
        except Exception as e:
            logging.error(f"BTTS tahmini hesaplanırken hata: {str(e)}")
            return 0.5  # Varsayılan değer

    def predict_cards(self, df: pd.DataFrame, home_team: str, away_team: str) -> Dict:
        """Kart tahminleri yap"""
        try:
            # Takımların agresiflik seviyeleri
            home_aggression = np.random.uniform(1.5, 3.5)  # 1.5-3.5 arası rastgele
            away_aggression = np.random.uniform(1.5, 3.5)  # 1.5-3.5 arası rastgele

            # Takımların son form durumlarına göre kart eğilimleri
        home_stats = self._get_team_stats(df, home_team)
        away_stats = self._get_team_stats(df, away_team)

            form_factor = ((home_stats['recent_form'] + away_stats['recent_form']) / 2) * \
                         np.random.uniform(0.8, 1.2)

            # Hakem faktörü (her hakem için farklı)
            referee_strictness = np.random.uniform(0.8, 1.4)
            
            # Derbi/önemli maç faktörü
            derby_factor = 1.3 if home_team in ["Fenerbahçe", "Galatasaray", "Beşiktaş", "Trabzonspor"] and \
                                 away_team in ["Fenerbahçe", "Galatasaray", "Beşiktaş", "Trabzonspor"] else 1.0
            
            # Maç önemi ve atmosfer faktörü
            match_intensity = np.random.uniform(0.9, 1.3)
            
            # Toplam kart tahmini
            base_cards = (home_aggression + away_aggression) * form_factor
            expected_cards = base_cards * referee_strictness * derby_factor * match_intensity
            
            # Alt/Üst olasılıkları için daha dinamik hesaplama
            mean_cards = expected_cards
            std_cards = expected_cards * 0.2  # Standart sapma
            
            over_prob = 1 - stats.norm.cdf(3.5, mean_cards, std_cards)
            over_prob = min(0.95, max(0.05, over_prob))  # Sınırla

        return {
                'expected_total': round(expected_cards, 1),
                'over_3.5_cards': over_prob,
                'under_3.5_cards': 1 - over_prob
            }
            
        except Exception as e:
            logging.error(f"Kart tahmini hatası: {str(e)}")
            return {'expected_total': 0, 'over_3.5_cards': 0, 'under_3.5_cards': 0}

    def predict_corners(self, df: pd.DataFrame, home_team: str, away_team: str) -> Dict:
        """Korner tahminleri yap"""
        try:
            # Takımların temel korner eğilimleri
            home_stats = self._get_team_stats(df, home_team)
            away_stats = self._get_team_stats(df, away_team)
            
            # Hücum stilleri (form ve gol ortalamasına göre)
            home_attack_style = np.random.uniform(0.7, 1.2) * (1 + home_stats['recent_form']) * \
                              (1 + home_stats['avg_goals_scored'] / 4)  # Azaltıldı
            away_attack_style = np.random.uniform(0.7, 1.2) * (1 + away_stats['recent_form']) * \
                              (1 + away_stats['avg_goals_scored'] / 4)  # Azaltıldı
            
            # Temel korner tahminleri (değerler düşürüldü)
            home_corners = np.random.normal(4, 1.2) * home_attack_style  # Ortalama ve std düşürüldü
            away_corners = np.random.normal(3, 1.2) * away_attack_style  # Ortalama ve std düşürüldü
            
            # Maç dinamikleri (faktörler düşürüldü)
            match_pace = np.random.uniform(0.7, 1.1)  # Maçın temposu
            tactical_factor = np.random.uniform(0.8, 1.1)  # Taktik yaklaşım
            weather_impact = np.random.uniform(0.9, 1.0)  # Hava durumu etkisi
            
            # Toplam korner tahmini
            expected_corners = (home_corners + away_corners) * match_pace * \
                             tactical_factor * weather_impact
            
            # Derbi faktörü (azaltıldı)
            if home_team in ["Fenerbahçe", "Galatasaray", "Beşiktaş", "Trabzonspor"] and \
               away_team in ["Fenerbahçe", "Galatasaray", "Beşiktaş", "Trabzonspor"]:
                expected_corners *= np.random.uniform(1.05, 1.2)  # Derbi etkisi azaltıldı
            
            # Üst sınır kontrolü
            expected_corners = min(expected_corners, 13.0)  # Maksimum korner sayısı sınırlandı
            
            # Alt/Üst olasılıkları için daha gerçekçi hesaplama
            mean_corners = expected_corners
            std_corners = expected_corners * 0.12  # Standart sapma azaltıldı
            
            # 9.5 korner için olasılık hesaplama
            over_prob = 1 - stats.norm.cdf(9.5, mean_corners, std_corners)
            over_prob = min(0.85, max(0.15, over_prob))  # Olasılık sınırları daraltıldı

        return {
                'expected_total': round(expected_corners, 1),
                'over_9.5_corners': over_prob,
                'under_9.5_corners': 1 - over_prob
            }
            
        except Exception as e:
            logging.error(f"Korner tahmini hatası: {str(e)}")
            return {'expected_total': 0, 'over_9.5_corners': 0, 'under_9.5_corners': 0}

class OddsBasedModel:
    def calculate_probabilities(self, home_odds: float, draw_odds: float, away_odds: float) -> np.ndarray:
        """Calculate probabilities from odds using margin removal"""
        if not all([home_odds, draw_odds, away_odds]):
            return None

        # Calculate margin
        margin = (1/float(home_odds) + 1/float(draw_odds) + 1/float(away_odds)) - 1

        # Calculate true probabilities
        p_home = (1/float(home_odds)) / (1 + margin)
        p_draw = (1/float(draw_odds)) / (1 + margin)
        p_away = (1/float(away_odds)) / (1 + margin)

        return np.array([p_home, p_draw, p_away])