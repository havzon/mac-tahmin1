import pandas as pd
import numpy as np
from typing import List, Dict, Optional

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

    def predict_with_player_performance(self, 
                                      match_stats: Dict, 
                                      events: List[Dict], 
                                      home_players: List[Dict], 
                                      away_players: List[Dict], 
                                      historical_data: Dict) -> Dict:
        """Oyuncu performanslarını dahil ederek tahmin yap"""
        try:
            # Temel olasılıkları hesapla
            base_probs = self.calculate_probabilities(
                historical_data.get('data', pd.DataFrame()),
                match_stats[0]['team']['name'],
                match_stats[1]['team']['name']
            )

            # Oyuncu performans analizleri
            home_performance = self._analyze_team_players(home_players, True)
            away_performance = self._analyze_team_players(away_players, False)

            # Performans faktörlerini hesapla
            home_factor = (
                home_performance['attack_strength'] * 0.4 +
                home_performance['midfield_control'] * 0.3 +
                home_performance['defense_stability'] * 0.3
            )

            away_factor = (
                away_performance['attack_strength'] * 0.4 +
                away_performance['midfield_control'] * 0.3 +
                away_performance['defense_stability'] * 0.3
            )

            # Form ve momentum etkisi
            home_form = self._calculate_team_form(match_stats[0], events)
            away_form = self._calculate_team_form(match_stats[1], events)

            # Olasılıkları güncelle
            adjusted_probs = self._adjust_probabilities(
                base_probs,
                home_factor,
                away_factor,
                home_form,
                away_form
            )

            return {
                'probabilities': adjusted_probs.tolist(),
                'player_analysis': {
                    'home_team': {
                        'key_players': self._identify_key_players(home_players),
                        'overall_rating': home_factor
                    },
                    'away_team': {
                        'key_players': self._identify_key_players(away_players),
                        'overall_rating': away_factor
                    }
                },
                'form_analysis': {
                    'home_team': home_form,
                    'away_team': away_form
                }
            }

        except Exception as e:
            print(f"Error in predict_with_player_performance: {str(e)}")
            return {
                'probabilities': base_probs.tolist(),
                'player_analysis': {'home_team': {}, 'away_team': {}},
                'form_analysis': {'home_team': {}, 'away_team': {}}
            }

    def _analyze_team_players(self, players: List[Dict], is_home: bool) -> Dict[str, float]:
        """Takım oyuncularının performansını analiz et"""
        try:
            if not players:
                return {'attack_strength': 0.5, 'midfield_control': 0.5, 'defense_stability': 0.5}

            # Oyuncuları pozisyonlarına göre grupla
            attackers = [p for p in players if p.get('position', '').lower() in ['forward', 'striker']]
            midfielders = [p for p in players if p.get('position', '').lower() in ['midfielder']]
            defenders = [p for p in players if p.get('position', '').lower() in ['defender']]

            # Her bölüm için performans hesapla
            attack_strength = self._calculate_line_performance(attackers, 'attack')
            midfield_control = self._calculate_line_performance(midfielders, 'midfield')
            defense_stability = self._calculate_line_performance(defenders, 'defense')

            # Ev sahibi avantajını hesaba kat
            if is_home:
                attack_strength *= 1.1
                midfield_control *= 1.05

            return {
                'attack_strength': min(1.0, attack_strength),
                'midfield_control': min(1.0, midfield_control),
                'defense_stability': min(1.0, defense_stability)
            }

        except Exception as e:
            print(f"Error in _analyze_team_players: {str(e)}")
            return {'attack_strength': 0.5, 'midfield_control': 0.5, 'defense_stability': 0.5}

    def _calculate_line_performance(self, players: List[Dict], line_type: str) -> float:
        """Belirli bir pozisyondaki oyuncuların performansını hesapla"""
        if not players:
            return 0.5

        metrics = {
            'attack': ['goals', 'shots_on_target', 'assists'],
            'midfield': ['passes_completed', 'chances_created', 'tackles'],
            'defense': ['tackles_won', 'interceptions', 'clearances']
        }

        total_score = 0
        for player in players:
            stats = player.get('statistics', {})
            score = sum(float(stats.get(metric, 0)) for metric in metrics[line_type])
            form = float(stats.get('form', 0.5))
            total_score += score * form

        return min(1.0, total_score / (len(players) * len(metrics[line_type])))

    def _calculate_team_form(self, team_stats: Dict, events: List[Dict]) -> Dict[str, float]:
        """Takım formunu hesapla"""
        try:
            recent_form = float(team_stats.get('form', 0.5))
            momentum = self._calculate_momentum(events, team_stats['team']['name'])

            return {
                'recent_form': recent_form,
                'momentum': momentum,
                'combined_score': (recent_form * 0.7 + momentum * 0.3)
            }

        except Exception as e:
            print(f"Error in _calculate_team_form: {str(e)}")
            return {'recent_form': 0.5, 'momentum': 0.5, 'combined_score': 0.5}

    def _calculate_momentum(self, events: List[Dict], team_name: str) -> float:
        """Maç içi momentumu hesapla"""
        if not events:
            return 0.5

        team_events = [e for e in events if e.get('team', {}).get('name') == team_name]
        if not team_events:
            return 0.5

        # Son olayların etkisini hesapla
        recent_impact = sum(self._get_event_impact(e) for e in team_events[-3:]) / 3
        return min(1.0, max(0.0, 0.5 + recent_impact))

    def _get_event_impact(self, event: Dict) -> float:
        """Olay etkisini hesapla"""
        impact_weights = {
            'Goal': 0.3,
            'Card': -0.1,
            'Substitution': 0.05,
            'Var': -0.05
        }
        return impact_weights.get(event.get('type', ''), 0)

    def _identify_key_players(self, players: List[Dict]) -> List[Dict]:
        """Kilit oyuncuları belirle"""
        try:
            key_players = []
            for player in players:
                stats = player.get('statistics', {})
                performance = {
                    'name': player.get('name', ''),
                    'position': player.get('position', ''),
                    'performance': {
                        'scoring_ability': float(stats.get('goals', 0)) / 10,
                        'passing_efficiency': float(stats.get('passes_completed', 0)) / 100,
                        'match_influence': float(stats.get('rating', 5)) / 10
                    }
                }

                # Performans skoru hesapla
                score = sum(performance['performance'].values()) / 3
                if score > 0.6:  # Sadece yüksek performanslı oyuncuları ekle
                    key_players.append(performance)

            return sorted(key_players, 
                        key=lambda x: sum(x['performance'].values()) / 3, 
                        reverse=True)[:3]  # En iyi 3 oyuncu

        except Exception as e:
            print(f"Error in _identify_key_players: {str(e)}")
            return []

    def _adjust_probabilities(self, base_probs: np.ndarray,
                            home_factor: float,
                            away_factor: float,
                            home_form: Dict[str, float],
                            away_form: Dict[str, float]) -> np.ndarray:
        """Olasılıkları performans faktörlerine göre ayarla"""
        try:
            # Form ve performans etkilerini hesapla
            home_adjustment = (home_factor * 0.6 + home_form['combined_score'] * 0.4 - 0.5) * 0.2
            away_adjustment = (away_factor * 0.6 + away_form['combined_score'] * 0.4 - 0.5) * 0.2

            # Olasılıkları güncelle
            adjusted_probs = base_probs.copy()
            adjusted_probs[0] += home_adjustment
            adjusted_probs[2] += away_adjustment
            adjusted_probs[1] = 1 - (adjusted_probs[0] + adjusted_probs[2])

            # Negatif olasılıkları düzelt
            adjusted_probs = np.maximum(adjusted_probs, 0)

            # Normalize et
            return adjusted_probs / adjusted_probs.sum()

        except Exception as e:
            print(f"Error in _adjust_probabilities: {str(e)}")
            return base_probs

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