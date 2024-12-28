import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import streamlit as st

class PredictionModel:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)  # Reduced trees, parallel processing
        self.xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)  # Parallel processing
        self.scaler = StandardScaler()

    def prepare_features(self, df, team1, team2):
        """Prepare features for prediction models"""
        features = []

        # Get last 5 matches for each team
        team1_matches = df[(df['HomeTeam'] == team1) | (df['AwayTeam'] == team1)].tail(5)
        team2_matches = df[(df['HomeTeam'] == team2) | (df['AwayTeam'] == team2)].tail(5)

        # Calculate form (wins in last 5 matches)
        team1_form = self._calculate_form(team1_matches, team1)
        team2_form = self._calculate_form(team2_matches, team2)

        # Calculate average goals scored and conceded
        team1_goals_scored = self._calculate_goals_scored(df, team1)
        team1_goals_conceded = self._calculate_goals_conceded(df, team1)
        team2_goals_scored = self._calculate_goals_scored(df, team2)
        team2_goals_conceded = self._calculate_goals_conceded(df, team2)

        features = [
            team1_form, team2_form,
            team1_goals_scored, team1_goals_conceded,
            team2_goals_scored, team2_goals_conceded
        ]

        return np.array(features).reshape(1, -1)

    def _calculate_form(self, matches, team):
        wins = 0
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team and match['FTR'] == 'H':
                wins += 1
            elif match['AwayTeam'] == team and match['FTR'] == 'A':
                wins += 1
        return wins / 5.0 if len(matches) > 0 else 0.0

    def _calculate_goals_scored(self, df, team):
        home_goals = df[df['HomeTeam'] == team]['FTHG'].mean()
        away_goals = df[df['AwayTeam'] == team]['FTAG'].mean()
        return (home_goals + away_goals) / 2 if not np.isnan(home_goals + away_goals) else 0.0

    def _calculate_goals_conceded(self, df, team):
        home_conceded = df[df['HomeTeam'] == team]['FTAG'].mean()
        away_conceded = df[df['AwayTeam'] == team]['FTHG'].mean()
        return (home_conceded + away_conceded) / 2 if not np.isnan(home_conceded + away_conceded) else 0.0

    def train(self, df):
        """Train both models on historical data"""
        X = []
        y = []

        # Show progress bar
        total_rows = len(df) - 5
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(len(df)-5):
            # Update progress
            progress = (i + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"Processing matches: {i+1}/{total_rows}")

            match = df.iloc[i]
            try:
                features = self.prepare_features(df.iloc[:i], match['HomeTeam'], match['AwayTeam'])
                X.append(features[0])

                # Convert result to numeric
                if match['FTR'] == 'H':
                    result = 0
                elif match['FTR'] == 'D':
                    result = 1
                else:
                    result = 2
                y.append(result)
            except Exception as e:
                st.warning(f"Skipping match due to data issue: {e}")
                continue

        X = np.array(X)
        y = np.array(y)

        # Scale features
        status_text.text("Scaling features...")
        X = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train models
        status_text.text("Training Random Forest model...")
        self.rf_model.fit(X_train, y_train)

        status_text.text("Training XGBoost model...")
        self.xgb_model.fit(X_train, y_train)

        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()

        st.success("Model training completed!")

    def predict(self, features):
        """Make predictions using both models"""
        features_scaled = self.scaler.transform(features)

        rf_pred = self.rf_model.predict_proba(features_scaled)[0]
        xgb_pred = self.xgb_model.predict_proba(features_scaled)[0]

        # Average predictions from both models
        combined_pred = (rf_pred + xgb_pred) / 2

        return combined_pred

class OddsBasedModel:
    def calculate_probabilities(self, home_odds, draw_odds, away_odds):
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

class StatisticalModel:
    def calculate_probabilities(self, df, home_team, away_team):
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
    
    def _get_team_stats(self, df, team):
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