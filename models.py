import pandas as pd
import numpy as np
from typing import List, Dict

class StatisticalModel:
    def __init__(self):
        """İstatistiksel tahmin modeli"""
        pass

    def calculate_probabilities(self, df: pd.DataFrame, home_team: str, away_team: str) -> np.ndarray:
        """Calculate probabilities based on historical statistics"""
        home_stats = self._get_team_stats(df, home_team)
        away_stats = self._get_team_stats(df, away_team)

        # Calculate base probabilities from historical win rates
        p_home = home_stats['home_win_rate'] * 0.6 + away_stats['away_loss_rate'] * 0.4
        p_away = away_stats['away_win_rate'] * 0.6 + home_stats['home_loss_rate'] * 0.4
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
        """Calculate team statistics from historical data"""
        home_matches = df[df['HomeTeam'] == team]
        away_matches = df[df['AwayTeam'] == team]

        stats = {
            'home_win_rate': len(home_matches[home_matches['FTR'] == 'H']) / len(home_matches) if len(home_matches) > 0 else 0.33,
            'away_win_rate': len(away_matches[away_matches['FTR'] == 'A']) / len(away_matches) if len(away_matches) > 0 else 0.33,
            'home_loss_rate': len(home_matches[home_matches['FTR'] == 'A']) / len(home_matches) if len(home_matches) > 0 else 0.33,
            'away_loss_rate': len(away_matches[away_matches['FTR'] == 'H']) / len(away_matches) if len(away_matches) > 0 else 0.33,
            'avg_goals_scored': (home_matches['FTHG'].mean() + away_matches['FTAG'].mean()) / 2,
            'avg_goals_conceded': (home_matches['FTAG'].mean() + away_matches['FTHG'].mean()) / 2
        }

        return stats

    def predict_goals(self, df: pd.DataFrame, home_team: str, away_team: str) -> float:
        """Predict expected goals in the match"""
        home_stats = self._get_team_stats(df, home_team)
        away_stats = self._get_team_stats(df, away_team)

        expected_home_goals = (home_stats['avg_goals_scored'] + away_stats['avg_goals_conceded']) / 2
        expected_away_goals = (away_stats['avg_goals_scored'] + home_stats['avg_goals_conceded']) / 2

        return expected_home_goals + expected_away_goals

    def predict_over_under(self, df: pd.DataFrame, home_team: str, away_team: str) -> List[float]:
        """Predict over/under 2.5 probabilities"""
        expected_goals = self.predict_goals(df, home_team, away_team)

        # Using Poisson distribution for over/under probabilities
        over_prob = 1 - np.exp(-expected_goals) * (1 + expected_goals + (expected_goals**2)/2)
        under_prob = 1 - over_prob

        return [over_prob, under_prob]

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