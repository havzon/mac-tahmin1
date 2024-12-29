import pandas as pd
import numpy as np
import math
from typing import List, Dict, Tuple

class StatisticalModel:
    def __init__(self):
        """İstatistiksel tahmin modeli"""
        pass

    def calculate_probabilities(self, df: pd.DataFrame, home_team: str, away_team: str) -> np.ndarray:
        """Calculate probabilities based on historical statistics"""
        home_stats = self._get_team_stats(df, home_team)
        away_stats = self._get_team_stats(df, away_team)

        # Calculate base probabilities from historical win rates with weighted recent form
        p_home = home_stats['home_win_rate'] * 0.4 + home_stats['recent_form'] * 0.3 + away_stats['away_loss_rate'] * 0.3
        p_away = away_stats['away_win_rate'] * 0.4 + away_stats['recent_form'] * 0.3 + home_stats['home_loss_rate'] * 0.3
        p_draw = 1 - (p_home + p_away)

        # Adjust probabilities based on goals scored/conceded
        goals_factor = (home_stats['avg_goals_scored'] - away_stats['avg_goals_conceded']) - \
                      (away_stats['avg_goals_scored'] - home_stats['avg_goals_conceded'])

        # Apply goals adjustment
        adjustment = goals_factor * 0.1
        p_home += adjustment
        p_away -= adjustment

        # Normalize probabilities
        total = p_home + p_draw + p_away
        return np.array([p_home/total, p_draw/total, p_away/total])

    def _get_team_stats(self, df: pd.DataFrame, team: str) -> Dict[str, float]:
        """Calculate team statistics from historical data with recent form"""
        # Son 10 maçı al
        recent_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(10)

        # Son 5 maç için form hesapla
        last_5_matches = recent_matches.tail(5)
        form_points = []

        for _, match in last_5_matches.iterrows():
            if match['HomeTeam'] == team:
                form_points.append(3 if match['FTR'] == 'H' else 1 if match['FTR'] == 'D' else 0)
            else:
                form_points.append(3 if match['FTR'] == 'A' else 1 if match['FTR'] == 'D' else 0)

        recent_form = sum(form_points) / (len(form_points) * 3) if form_points else 0.33

        home_matches = df[df['HomeTeam'] == team]
        away_matches = df[df['AwayTeam'] == team]

        stats = {
            'home_win_rate': len(home_matches[home_matches['FTR'] == 'H']) / len(home_matches) if len(home_matches) > 0 else 0.33,
            'away_win_rate': len(away_matches[away_matches['FTR'] == 'A']) / len(away_matches) if len(away_matches) > 0 else 0.33,
            'home_loss_rate': len(home_matches[home_matches['FTR'] == 'A']) / len(home_matches) if len(home_matches) > 0 else 0.33,
            'away_loss_rate': len(away_matches[away_matches['FTR'] == 'H']) / len(away_matches) if len(away_matches) > 0 else 0.33,
            'avg_goals_scored': (home_matches['FTHG'].mean() + away_matches['FTAG'].mean()) / 2,
            'avg_goals_conceded': (home_matches['FTAG'].mean() + away_matches['FTHG'].mean()) / 2,
            'recent_form': recent_form
        }

        return stats

    def predict_goals(self, df: pd.DataFrame, home_team: str, away_team: str) -> Tuple[float, List[float]]:
        """Predict expected goals and over/under probabilities"""
        home_stats = self._get_team_stats(df, home_team)
        away_stats = self._get_team_stats(df, away_team)

        # Expected goals calculation with form adjustment
        home_form_factor = (home_stats['recent_form'] - 0.33) * 0.5
        away_form_factor = (away_stats['recent_form'] - 0.33) * 0.5

        expected_home_goals = (home_stats['avg_goals_scored'] + away_stats['avg_goals_conceded']) / 2 + home_form_factor
        expected_away_goals = (away_stats['avg_goals_scored'] + home_stats['avg_goals_conceded']) / 2 + away_form_factor

        total_expected_goals = expected_home_goals + expected_away_goals

        # Calculate over/under probabilities using Poisson distribution
        probs = []
        for line in [0.5, 1.5, 2.5, 3.5]:
            over_prob = 1 - np.exp(-total_expected_goals) * sum([(total_expected_goals**k) / math.factorial(k) for k in range(int(line)+1)])
            probs.append(over_prob)

        return total_expected_goals, probs

    def predict_first_half_goals(self, df: pd.DataFrame, home_team: str, away_team: str) -> Tuple[float, List[float]]:
        """İlk yarı gol tahmini yap"""
        home_matches = df[df['HomeTeam'] == home_team]
        away_matches = df[df['AwayTeam'] == away_team]

        # İlk yarı gol ortalamaları
        home_first_half_scored = home_matches['HTHG'].mean()
        away_first_half_scored = away_matches['HTAG'].mean()

        expected_first_half_goals = (home_first_half_scored + away_first_half_scored) / 2

        # İlk yarı üst/alt olasılıkları
        probs = []
        for line in [0.5, 1.5]:
            over_prob = 1 - np.exp(-expected_first_half_goals) * sum([(expected_first_half_goals**k) / math.factorial(k) for k in range(int(line)+1)])
            probs.append(over_prob)

        return expected_first_half_goals, probs

    def predict_both_teams_to_score(self, df: pd.DataFrame, home_team: str, away_team: str) -> float:
        """Karşılıklı gol olasılığını hesapla"""
        home_stats = self._get_team_stats(df, home_team)
        away_stats = self._get_team_stats(df, away_team)

        home_scoring_prob = (home_stats['avg_goals_scored'] / 2) * (1 + home_stats['recent_form'])
        away_scoring_prob = (away_stats['avg_goals_scored'] / 2) * (1 + away_stats['recent_form'])

        btts_prob = home_scoring_prob * away_scoring_prob
        return min(0.95, max(0.05, btts_prob))

    def predict_cards(self, df: pd.DataFrame, home_team: str, away_team: str) -> Dict[str, float]:
        """Kart tahminlerini yap"""
        # Varsayalım ki df'de 'HomeCards' ve 'AwayCards' sütunları var
        home_matches = df[df['HomeTeam'] == home_team]
        away_matches = df[df['AwayTeam'] == away_team]

        avg_home_cards = home_matches['HomeCards'].mean() if 'HomeCards' in df.columns else 2.0
        avg_away_cards = away_matches['AwayCards'].mean() if 'AwayCards' in df.columns else 2.0

        total_cards_exp = avg_home_cards + avg_away_cards

        return {
            'under_3.5_cards': 1 - (1 - np.exp(-total_cards_exp)),
            'over_3.5_cards': 1 - np.exp(-total_cards_exp),
            'expected_total': total_cards_exp
        }

    def predict_corners(self, df: pd.DataFrame, home_team: str, away_team: str) -> Dict[str, float]:
        """Korner tahminlerini yap"""
        # Varsayalım ki df'de 'HomeCorners' ve 'AwayCorners' sütunları var
        home_matches = df[df['HomeTeam'] == home_team]
        away_matches = df[df['AwayTeam'] == away_team]

        avg_home_corners = home_matches['HomeCorners'].mean() if 'HomeCorners' in df.columns else 5.0
        avg_away_corners = away_matches['AwayCorners'].mean() if 'AwayCorners' in df.columns else 4.0

        total_corners_exp = avg_home_corners + avg_away_corners

        return {
            'under_9.5_corners': 1 - (1 - np.exp(-total_corners_exp)),
            'over_9.5_corners': 1 - np.exp(-total_corners_exp),
            'expected_total': total_corners_exp
        }

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