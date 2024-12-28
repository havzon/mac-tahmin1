import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import streamlit as st

class PredictionModel:
    def __init__(self):
        # Model parametrelerini optimize ettim
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()

    def prepare_features(self, df, team1, team2):
        """Prepare features for prediction models"""
        features = []

        # Son 10 maç verisi
        team1_matches = df[(df['HomeTeam'] == team1) | (df['AwayTeam'] == team1)].tail(10)
        team2_matches = df[(df['HomeTeam'] == team2) | (df['AwayTeam'] == team2)].tail(10)

        # Form metriklerini hesapla
        team1_form = self.get_team_form(df, team1)
        team2_form = self.get_team_form(df, team2)

        # Son 5 maçtaki gol ortalamaları
        team1_recent = df[(df['HomeTeam'] == team1) | (df['AwayTeam'] == team1)].tail(5)
        team2_recent = df[(df['HomeTeam'] == team2) | (df['AwayTeam'] == team2)].tail(5)

        team1_recent_goals = self._calculate_recent_goals(team1_recent, team1)
        team2_recent_goals = self._calculate_recent_goals(team2_recent, team2)

        # Genel gol istatistikleri
        team1_goals_scored = self._calculate_goals_scored(df, team1)
        team1_goals_conceded = self._calculate_goals_conceded(df, team1)
        team2_goals_scored = self._calculate_goals_scored(df, team2)
        team2_goals_conceded = self._calculate_goals_conceded(df, team2)

        # Head-to-head istatistikleri
        h2h_stats = self._calculate_h2h_stats(df, team1, team2)

        features = [
            *team1_form,  # 5 form metriği
            *team2_form,  # 5 form metriği
            team1_recent_goals['scored'],  # Son 5 maçtaki gol ortalaması
            team1_recent_goals['conceded'],
            team2_recent_goals['scored'],
            team2_recent_goals['conceded'],
            team1_goals_scored,  # Genel gol ortalaması
            team1_goals_conceded,
            team2_goals_scored,
            team2_goals_conceded,
            h2h_stats['team1_wins'],  # Head-to-head istatistikleri
            h2h_stats['team2_wins'],
            h2h_stats['draws']
        ]

        return np.array(features).reshape(1, -1)

    def _calculate_recent_goals(self, matches, team):
        """Son maçlardaki gol istatistiklerini hesapla"""
        goals_scored = 0
        goals_conceded = 0
        num_matches = len(matches)

        if num_matches == 0:
            return {'scored': 0, 'conceded': 0}

        for _, match in matches.iterrows():
            if match['HomeTeam'] == team:
                goals_scored += match['FTHG']
                goals_conceded += match['FTAG']
            else:
                goals_scored += match['FTAG']
                goals_conceded += match['FTHG']

        return {
            'scored': goals_scored / num_matches,
            'conceded': goals_conceded / num_matches
        }

    def _calculate_h2h_stats(self, df, team1, team2):
        """Head-to-head istatistiklerini hesapla"""
        h2h_matches = df[
            ((df['HomeTeam'] == team1) & (df['AwayTeam'] == team2)) |
            ((df['HomeTeam'] == team2) & (df['AwayTeam'] == team1))
        ].tail(5)

        if len(h2h_matches) == 0:
            return {'team1_wins': 0.33, 'team2_wins': 0.33, 'draws': 0.34}

        team1_wins = 0
        team2_wins = 0
        draws = 0

        for _, match in h2h_matches.iterrows():
            if match['HomeTeam'] == team1:
                if match['FTR'] == 'H':
                    team1_wins += 1
                elif match['FTR'] == 'A':
                    team2_wins += 1
                else:
                    draws += 1
            else:
                if match['FTR'] == 'H':
                    team2_wins += 1
                elif match['FTR'] == 'A':
                    team1_wins += 1
                else:
                    draws += 1

        num_matches = len(h2h_matches)
        return {
            'team1_wins': team1_wins / num_matches,
            'team2_wins': team2_wins / num_matches,
            'draws': draws / num_matches
        }

    def _calculate_goals_scored(self, df, team):
        recent_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(10)
        if len(recent_matches) == 0:
            return 0.0

        home_goals = recent_matches[recent_matches['HomeTeam'] == team]['FTHG'].mean()
        away_goals = recent_matches[recent_matches['AwayTeam'] == team]['FTAG'].mean()

        if np.isnan(home_goals):
            home_goals = 0
        if np.isnan(away_goals):
            away_goals = 0

        return (home_goals + away_goals) / 2

    def _calculate_goals_conceded(self, df, team):
        recent_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(10)
        if len(recent_matches) == 0:
            return 0.0

        home_conceded = recent_matches[recent_matches['HomeTeam'] == team]['FTAG'].mean()
        away_conceded = recent_matches[recent_matches['AwayTeam'] == team]['FTHG'].mean()

        if np.isnan(home_conceded):
            home_conceded = 0
        if np.isnan(away_conceded):
            away_conceded = 0

        return (home_conceded + away_conceded) / 2

    def train(self, df):
        """Train both models on historical data using recent matches"""
        # Son 1000 maçı kullan
        recent_df = df.tail(1000)

        X = []
        y = []

        total_rows = len(recent_df) - 5
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(len(recent_df)-5):
            progress = (i + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"İşlenen maç: {i+1}/{total_rows}")

            match = recent_df.iloc[i]
            try:
                features = self.prepare_features(recent_df.iloc[:i], match['HomeTeam'], match['AwayTeam'])
                X.append(features[0])

                # Sonucu sayısal değere dönüştür
                if match['FTR'] == 'H':
                    result = 0
                elif match['FTR'] == 'D':
                    result = 1
                else:
                    result = 2
                y.append(result)
            except Exception as e:
                continue

        X = np.array(X)
        y = np.array(y)

        # Özellikleri ölçeklendir
        status_text.text("Özellikler ölçeklendiriliyor...")
        X = self.scaler.fit_transform(X)

        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modelleri eğit
        status_text.text("Random Forest modeli eğitiliyor...")
        self.rf_model.fit(X_train, y_train)

        status_text.text("XGBoost modeli eğitiliyor...")
        self.xgb_model.fit(X_train, y_train)

        # İlerleme göstergelerini temizle
        progress_bar.empty()
        status_text.empty()

        st.success("Model eğitimi tamamlandı!")

    def predict(self, features):
        """Her iki modeli kullanarak tahmin yap"""
        features_scaled = self.scaler.transform(features)

        # Model tahminleri
        rf_pred = self.rf_model.predict_proba(features_scaled)[0]
        xgb_pred = self.xgb_model.predict_proba(features_scaled)[0]

        # Tahminleri birleştir (ağırlıklı ortalama)
        combined_pred = (rf_pred * 0.6 + xgb_pred * 0.4)  # RF modeline daha fazla ağırlık ver

        # Ev sahibi avantajını hesaba kat
        home_advantage = 0.1
        combined_pred[0] += home_advantage  # Ev sahibi kazanma olasılığını artır
        combined_pred[2] -= home_advantage  # Deplasman kazanma olasılığını azalt

        # Olasılıkları normalize et
        total = sum(combined_pred)
        combined_pred = combined_pred / total

        return combined_pred

    def predict_goals(self, features):
        """Maçtaki toplam gol sayısını tahmin et"""
        home_goals = features[0][10]  # Son 5 maçtaki ev sahibi gol ortalaması
        away_goals = features[0][12]  # Son 5 maçtaki deplasman gol ortalaması
        home_conceded = features[0][11]  # Ev sahibinin yediği goller
        away_conceded = features[0][13]  # Deplasman yediği goller

        # Beklenen golleri hesapla
        expected_home_goals = (home_goals + away_conceded) / 2
        expected_away_goals = (away_goals + home_conceded) / 2

        # Form faktörünü ekle
        home_form = sum(features[0][:5]) / 5  # Ev sahibi form
        away_form = sum(features[0][5:10]) / 5  # Deplasman form

        expected_home_goals *= (1 + (home_form - 0.5) * 0.2)
        expected_away_goals *= (1 + (away_form - 0.5) * 0.2)

        return expected_home_goals + expected_away_goals

    def predict_over_under(self, features):
        """2.5 üst/alt olasılıklarını tahmin et"""
        expected_goals = self.predict_goals(features)

        # Poisson dağılımı kullanarak olasılıkları hesapla
        over_prob = 1 - np.exp(-expected_goals) * (1 + expected_goals + (expected_goals**2)/2)
        under_prob = 1 - over_prob

        # Form bazlı düzeltme
        total_form = (sum(features[0][:5]) + sum(features[0][5:10])) / 10
        if total_form > 0.6:  # İyi formda takımlar
            over_prob *= 1.1
            under_prob *= 0.9
        elif total_form < 0.4:  # Kötü formda takımlar
            over_prob *= 0.9
            under_prob *= 1.1

        # Olasılıkları normalize et
        total = over_prob + under_prob
        over_prob /= total
        under_prob /= total

        return [over_prob, under_prob]

    def get_team_form(self, df, team):
        """Kapsamlı takım form metrikleri hesapla"""
        recent_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(5)

        if len(recent_matches) == 0:
            return [0, 0, 0, 0, 0]

        wins = 0
        goals_scored = 0
        clean_sheets = 0
        goal_diff = 0
        points = 0

        for _, match in recent_matches.iterrows():
            if match['HomeTeam'] == team:
                goals_for = match['FTHG']
                goals_against = match['FTAG']
                if match['FTR'] == 'H':
                    wins += 1
                    points += 3
                elif match['FTR'] == 'D':
                    points += 1
            else:
                goals_for = match['FTAG']
                goals_against = match['FTHG']
                if match['FTR'] == 'A':
                    wins += 1
                    points += 3
                elif match['FTR'] == 'D':
                    points += 1

            goals_scored += goals_for
            if goals_against == 0:
                clean_sheets += 1
            goal_diff += goals_for - goals_against

        num_matches = len(recent_matches)
        max_points = num_matches * 3

        form_metrics = [
            wins / num_matches,  # Galibiyet oranı
            goals_scored / (num_matches * 3),  # Gol ortalaması (normalize edilmiş)
            clean_sheets / num_matches,  # Gol yememe oranı
            points / max_points,  # Puan bazlı form
            (goal_diff + 5) / 10  # Gol farkı (normalize edilmiş)
        ]

        return [min(max(x, 0), 1) for x in form_metrics]


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