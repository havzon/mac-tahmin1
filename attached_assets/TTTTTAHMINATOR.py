import requests
import json
import os
from datetime import date, datetime, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sqlite3
import sys
from datetime import datetime


API_KEY = "c417cb5967msh54c12f850f3798cp12a733jsn315fab53ee3c"
BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

headers = {
    'x-rapidapi-host': 'api-football-v1.p.rapidapi.com',
    'x-rapidapi-key': 'c417cb5967msh54c12f850f3798cp12a733jsn315fab53ee3c'
}

def save_analysis_results(analysis_results):
    # Şu anki tarih ve saati al
    current_time = datetime.datetime.now()
    
    # Dosya adı için tarih ve saat formatını belirle
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    
    # Sonuçları kaydedeceğimiz klasörü belirlen
    results_folder = "analysis_results"
    
    # Eğer klasör yoksa oluştur
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Dosya adını oluştur
    filename = f"analysis_results_{timestamp}.json"
    
    # Tam dosya yolunu oluştur
    file_path = os.path.join(results_folder, filename)
    
    # Sonuçları JSON formatında dosyaya kaydet
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {file_path}")

def is_time_in_range(match_time, start_time, end_time):
    match_time = datetime.strptime(match_time, "%H:%M").time()
    return start_time <= match_time <= end_time

def get_fixtures(date, start_time, end_time):
    url = f"{BASE_URL}/fixtures"
    querystring = {"date": date}
    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        all_fixtures = response.json()['response']
        filtered_fixtures = [
            fixture for fixture in all_fixtures
            if is_time_in_range(fixture['fixture']['date'].split('T')[1][:5], start_time, end_time)
        ]
        if not filtered_fixtures:
            print(f"No fixtures found on {date} between {start_time} and {end_time}")
        return filtered_fixtures
    except requests.exceptions.RequestException as e:
        print(f"Error fetching fixtures: {e}")
        return []

def get_odds(fixture_id):
    url = f"{BASE_URL}/odds"
    querystring = {"fixture": fixture_id}
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200 and response.json()['response']:
        return response.json()['response'][0]
    else:
        print(f"Failed to get odds for fixture {fixture_id}. Status code: {response.status_code}")
        return None

def get_team_statistics(team_id, league_id, season):
    url = f"{BASE_URL}/teams/statistics"
    querystring = {"team": team_id, "league": league_id, "season": season}
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json()['response']
    else:
        print(f"Failed to get team statistics. Status code: {response.status_code}")
        return None

def get_h2h(team1_id, team2_id):
    url = f"{BASE_URL}/fixtures/headtohead"
    querystring = {"h2h": f"{team1_id}-{team2_id}"}
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json()['response']
    else:
        print(f"Failed to get H2H data. Status code: {response.status_code}")
        return []

def poisson_probability(lambda_, k):
    return stats.poisson.pmf(k, lambda_)

def get_team_ratings(fixture):
    """
    Fetches the current ratings for the teams involved in a fixture from the API-Football API.
    
    Args:
        fixture (dict): A dictionary containing information about the fixture.
    
    Returns:
        tuple: A tuple containing the home team's rating and the away team's rating.
    """
    headers = {
        'X-RapidAPI-Key': 'YOUR_API_KEY_HERE',
        'X-RapidAPI-Host': 'api-football-v1.p.rapidapi.com'
    }
    
    home_team_id = fixture['teams']['home']['id']
    away_team_id = fixture['teams']['away']['id']
    
    try:
        url = f"https://api-football-v1.p.rapidapi.com/v3/teams?id={home_team_id}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        home_team_data = json.loads(response.text)['response'][0]
        home_team_rating = home_team_data['power_rank']
        
        url = f"https://api-football-v1.p.rapidapi.com/v3/teams?id={away_team_id}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        away_team_data = json.loads(response.text)['response'][0]
        away_team_rating = away_team_data['power_rank']
        
        return home_team_rating, away_team_rating
    
    except requests.exceptions.RequestException as e:
        print(f"API bağlantısında hata oluştu: {e}")
        return 1500, 1500

# Örnek kullanım
fixture = {
    "teams": {
        "home": {
            "id": 1
        },
        "away": {
            "id": 2
        }
    }
}

home_rating, away_rating = get_team_ratings(fixture)
print(f"Home team rating: {home_rating}")
print(f"Away team rating: {away_rating}")

def kelly_criterion(probability, odds):
    q = 1 - probability
    b = odds - 1
    return (b * probability - q) / b if b > 0 else 0

def calculate_value(probability, odds):
    # Value hesaplamasına üst limit ekle
    max_value = 50.0
    value = (probability * (odds - 1)) - (1 - probability)
    return max(min(value * 100, max_value), -max_value)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def normalize_expected_goals(goals):
    return max(0.1, min(goals, 5.0))

def calculate_form_forecast(current_form, historical_data):
    # Ani değişimleri sınırla
    max_change = 0.25
    forecast = enhanced_time_series_analysis(historical_data)
    change = forecast - current_form
    return current_form + max(min(change, max_change), -max_change)

def calculate_form(form_data):
    if isinstance(form_data, str):
        recent_results = form_data[-5:]  # Son 5 maç
        return sum(1 for result in recent_results if result == 'W') / len(recent_results)
    elif isinstance(form_data, list):
        recent_results = form_data[-5:]  # Son 5 maç
        return sum(1 for result in recent_results if result == 'W') / len(recent_results)
    elif isinstance(form_data, dict) and 'wins' in form_data and 'played' in form_data:
        total_matches = form_data['played']['total']
        wins = form_data['wins']['total']
        return wins / total_matches if total_matches > 0 else 0.5
    else:
        print(f"Unexpected form data format: {form_data}")
        return 0.5  # default value

def time_series_analysis(historical_data):
    model = ARIMA(historical_data, order=(1, 1, 1))
    results = model.fit()
    forecast = results.forecast(steps=1)
    return forecast[0]

def lstm_time_series_analysis(historical_data, forecast_steps=1):
    data = np.array(historical_data).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(data_scaled) - forecast_steps):
        X.append(data_scaled[i:i+forecast_steps])
        y.append(data_scaled[i+forecast_steps])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    X, y = torch.FloatTensor(X), torch.FloatTensor(y)
    
    model = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
    linear = nn.Linear(50, 1)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(model.parameters()) + list(linear.parameters()), lr=0.001)
    
    for epoch in range(100):
        optimizer.zero_grad()
        lstm_out, _ = model(X)
        outputs = linear(lstm_out[:, -1, :])
        loss = criterion(outputs.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()
    
    last_sequence = torch.FloatTensor(data_scaled[-forecast_steps:]).view(1, forecast_steps, 1)
    with torch.no_grad():
        lstm_out, _ = model(last_sequence)
        next_prediction = linear(lstm_out[:, -1, :])
    return scaler.inverse_transform(next_prediction.numpy())[0, 0]

def prophet_time_series_analysis(historical_data, forecast_steps=1):
    df = pd.DataFrame({'ds': pd.date_range(end=pd.Timestamp.today(), periods=len(historical_data)),
                       'y': historical_data})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_steps)
    forecast = model.predict(future)
    return forecast['yhat'].iloc[-1]

def enhanced_time_series_analysis(historical_data):
    lstm_forecast = lstm_time_series_analysis(historical_data)
    prophet_forecast = prophet_time_series_analysis(historical_data)
    arima_forecast = time_series_analysis(historical_data)
    
    combined_forecast = (lstm_forecast + prophet_forecast + arima_forecast) / 3
    return combined_forecast

def bayes_network_analysis(team_stats):
    G = nx.DiGraph()
    G.add_edge("Attack", "Goals")
    G.add_edge("Defense", "Goals Against")
    G.add_edge("Form", "Performance")
    G.add_edge("Goals", "Performance")
    G.add_edge("Goals Against", "Performance")
    
    try:
        attack_strength = float(team_stats['goals']['for']['average']['total'])
    except (KeyError, TypeError, ValueError):
        attack_strength = 1.0
    
    try:
        defense_strength = float(team_stats['goals']['against']['average']['total'])
    except (KeyError, TypeError, ValueError):
        defense_strength = 1.0
    
    form = calculate_form(team_stats.get('form', []))
    
    performance_prob = (attack_strength * 0.4 + (1 / max(defense_strength, 0.1)) * 0.4 + form * 0.2)
    return min(performance_prob, 1.0)

def dixon_coles_model(home_goals, away_goals, home_attack, away_attack, home_defense, away_defense):
    lambda_home = home_attack * away_defense
    lambda_away = away_attack * home_defense
    
    p = poisson_probability(lambda_home, home_goals)
    q = poisson_probability(lambda_away, away_goals)
    
    if home_goals == 0 and away_goals == 0:
        correction = 1 - lambda_home * lambda_away / ((1 + lambda_home) * (1 + lambda_away))
    elif home_goals == 0 and away_goals == 1:
        correction = 1 + lambda_home / (1 + lambda_home)
    elif home_goals == 1 and away_goals == 0:
        correction = 1 + lambda_away / (1 + lambda_away)
    elif home_goals == 1 and away_goals == 1:
        correction = 1 - 1 / ((1 + lambda_home) * (1 + lambda_away))
    else:
        correction = 1
    
    return p * q * correction

def simple_bayesian_analysis(home_attack, away_attack, home_defense, away_defense, num_simulations=1000):
    home_goals = stats.poisson(home_attack * away_defense).rvs(num_simulations)
    away_goals = stats.poisson(away_attack * home_defense).rvs(num_simulations)
    
    home_win_prob = np.mean(home_goals > away_goals)
    draw_prob = np.mean(home_goals == away_goals)
    away_win_prob = np.mean(home_goals < away_goals)
    
    return home_win_prob, draw_prob, away_win_prob

def simulate_match(lambda_home, lambda_away, home_elo, away_elo, home_form, away_form, home_performance, away_performance, num_simulations=10000):
    elo_diff = home_elo - away_elo
    form_diff = home_form - away_form
    performance_diff = home_performance - away_performance

    home_advantage = 0.1

    adjusted_lambda_home = lambda_home * (1 + home_advantage) * (1 + 0.1 * elo_diff / 400) * (1 + 0.1 * form_diff) * (1 + 0.1 * performance_diff)
    adjusted_lambda_away = lambda_away * (1 - 0.1 * elo_diff / 400) * (1 - 0.1 * form_diff) * (1 - 0.1 * performance_diff)

    home_wins = 0
    draws = 0
    away_wins = 0
    total_goals = 0
    both_teams_score = 0
    home_wins_ht = 0
    draws_ht = 0
    away_wins_ht = 0

    for _ in range(num_simulations):
        home_score_ht = np.random.poisson(adjusted_lambda_home / 2)
        away_score_ht = np.random.poisson(adjusted_lambda_away / 2)
        home_score_ft = home_score_ht + np.random.poisson(adjusted_lambda_home / 2)
        away_score_ft = away_score_ht + np.random.poisson(adjusted_lambda_away / 2)

        if home_score_ft > away_score_ft:
            home_wins += 1
        elif home_score_ft == away_score_ft:
            draws += 1
        else:
            away_wins += 1

        total_goals += home_score_ft + away_score_ft

        if home_score_ft > 0 and away_score_ft > 0:
            both_teams_score += 1

        if home_score_ht > away_score_ht:
            home_wins_ht += 1
        elif home_score_ht == away_score_ht:
            draws_ht += 1
        else:
            away_wins_ht += 1

    results = {
        "home_win_prob": home_wins / num_simulations,
        "draw_prob": draws / num_simulations,
        "away_win_prob": away_wins / num_simulations,
        "avg_total_goals": total_goals / num_simulations,
        "both_teams_score_prob": both_teams_score / num_simulations,
        "over_2_5_prob": sum(np.random.poisson(adjusted_lambda_home) + np.random.poisson(adjusted_lambda_away) > 2.5 for _ in range(num_simulations)) / num_simulations,
        "home_win_ht_prob": home_wins_ht / num_simulations,
        "draw_ht_prob": draws_ht / num_simulations,
        "away_win_ht_prob": away_wins_ht / num_simulations,
    }

    return results

def calculate_final_probabilities(home_elo, away_elo, home_odds, draw_odds, away_odds, home_form, away_form, home_performance, away_performance, league_level, match_importance):
    elo_diff = home_elo - away_elo
    elo_prob_home = 1 / (1 + 10 ** (-elo_diff / 400))
    elo_prob_away = 1 - elo_prob_home

    total_prob = (1 / home_odds) + (1 / draw_odds) + (1 / away_odds)
    odds_prob_home = (1 / home_odds) / total_prob
    odds_prob_draw = (1 / draw_odds) / total_prob
    odds_prob_away = (1 / away_odds) / total_prob

    form_performance_home = (home_form + home_performance) / 2
    form_performance_away = (away_form + away_performance) / 2

    importance_factor = (league_level + match_importance) / 2

    # Ağırlıkları güncelle
    w_elo = 0.2
    w_odds = 0.5  
    w_form = 0.2
    w_importance = 0.1

    final_prob_home = (w_elo * elo_prob_home + 
                       w_odds * odds_prob_home + 
                       w_form * form_performance_home + 
                       w_importance * importance_factor)

    final_prob_away = (w_elo * elo_prob_away + 
                       w_odds * odds_prob_away + 
                       w_form * form_performance_away + 
                       w_importance * importance_factor)

    final_prob_draw = 1 - final_prob_home - final_prob_away
    
    # Olasılıkları normalize et
    total_prob = final_prob_home + final_prob_draw + final_prob_away
    final_prob_home /= total_prob
    final_prob_draw /= total_prob
    final_prob_away /= total_prob

    return final_prob_home, final_prob_draw, final_prob_away

def create_neural_network(input_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
        nn.Softmax(dim=1)
    )
    return model


def enhanced_ml_prediction(historical_data, current_features):
    try:
        # Veri boyutlarının kontrolü
        if len(historical_data) < 10:  # Minimum veri kontrolü
            return None
            
        # Veri hazırlama
        X = np.array(historical_data)
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # 2D array'e dönüştür
            
        # Current features'ın boyut kontrolü
        current_features = np.array(current_features).reshape(1, -1)
        
        # Veri standardizasyonu
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Model eğitimi
        model = RandomForestClassifier(n_estimators=100)
        
        # Train-test split
        X_train, X_test = train_test_split(X_scaled, test_size=0.2, shuffle=False)
        
        # Model fit ve tahmin
        model.fit(X_train, np.zeros(len(X_train)))  # Dummy target
        prediction = model.predict_proba(current_features)
        
        return prediction[0]
        
    except Exception as e:
        return f"Error in enhanced ML: {str(e)}"
    
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=10, n_jobs=-1)
    rf_grid.fit(X_train, y_train)

    # PyTorch model
    class CustomDataset(Dataset):
        def __init__(self, features, labels):
            self.features = torch.FloatTensor(features)
            self.labels = torch.LongTensor(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = create_neural_network(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        rf_pred = rf_grid.predict_proba(X_test)
        nn_pred = model(torch.FloatTensor(X_test)).numpy()
        combined_pred = (rf_pred + nn_pred) / 2

    return combined_pred

def feature_engineering(features):
    features_engineered = features.copy()
    if 'home_goals' in features and 'away_goals' in features:
        features_engineered['goal_diff'] = features['home_goals'] - features['away_goals']
        features_engineered['total_goals'] = features['home_goals'] + features['away_goals']
    if 'home_elo' in features and 'away_elo' in features:
        features_engineered['elo_diff'] = features['home_elo'] - features['away_elo']
    return features_engineered

def analyze_match(fixture, odds, home_stats, away_stats, h2h, league_data):
    result = {
        "fixture_id": fixture['fixture']['id'],
        "home_team": fixture['teams']['home']['name'],
        "away_team": fixture['teams']['away']['name'],
    }

    # Varsayılan değerler atayalım
    home_goals = away_goals = home_goals_against = away_goals_against = 0.0
    home_odds = draw_odds = away_odds = 2.0
    home_value = draw_value = away_value = 0.0

    try:
        home_goals = float(home_stats['goals']['for']['average']['total'])
        away_goals = float(away_stats['goals']['for']['average']['total'])
        home_goals_against = float(home_stats['goals']['against']['average']['total'])
        away_goals_against = float(away_stats['goals']['against']['average']['total'])
        print(f"Home goals: {home_goals}, Away goals: {away_goals}")
        
        result.update({
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_goals_against": home_goals_against,
            "away_goals_against": away_goals_against,
        })
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error processing goal statistics: {e}")
        result["error_goals"] = str(e)

    try:
        home_odds = float(next(bet for bet in odds['bookmakers'][0]['bets'] if bet['name'] == 'Match Winner')['values'][0]['odd'])
        draw_odds = float(next(bet for bet in odds['bookmakers'][0]['bets'] if bet['name'] == 'Match Winner')['values'][1]['odd'])
        away_odds = float(next(bet for bet in odds['bookmakers'][0]['bets'] if bet['name'] == 'Match Winner')['values'][2]['odd'])
        
        result.update({
            "home_odds": home_odds,
            "draw_odds": draw_odds,
            "away_odds": away_odds,
        })

        # Bahis oranlarından olasılık hesapla
        total_prob = (1 / home_odds) + (1 / draw_odds) + (1 / away_odds)
        odds_prob_home = (1 / home_odds) / total_prob
        odds_prob_draw = (1 / draw_odds) / total_prob
        odds_prob_away = (1 / away_odds) / total_prob

        result.update({
            "odds_implied_prob_home": f"{odds_prob_home:.2%}",
            "odds_implied_prob_draw": f"{odds_prob_draw:.2%}",
            "odds_implied_prob_away": f"{odds_prob_away:.2%}",
        })

    except (IndexError, StopIteration, KeyError, TypeError, ValueError) as e:
        print(f"Error processing odds: {e}")
        result["error_odds"] = str(e)

    try:
        home_odds = float(next(bet for bet in odds['bookmakers'][0]['bets'] if bet['name'] == 'Match Winner')['values'][0]['odd'])
        draw_odds = float(next(bet for bet in odds['bookmakers'][0]['bets'] if bet['name'] == 'Match Winner')['values'][1]['odd'])
        away_odds = float(next(bet for bet in odds['bookmakers'][0]['bets'] if bet['name'] == 'Match Winner')['values'][2]['odd'])
        
        result.update({
            "home_odds": home_odds,
            "draw_odds": draw_odds,
            "away_odds": away_odds,
        })

        # Bahis oranlarından olasılık hesapla
        total_prob = (1 / home_odds) + (1 / draw_odds) + (1 / away_odds)
        odds_prob_home = (1 / home_odds) / total_prob
        odds_prob_draw = (1 / draw_odds) / total_prob
        odds_prob_away = (1 / away_odds) / total_prob

        result.update({
            "odds_implied_prob_home": f"{odds_prob_home:.2%}",
            "odds_implied_prob_draw": f"{odds_prob_draw:.2%}",
            "odds_implied_prob_away": f"{odds_prob_away:.2%}",
        })

    except (IndexError, StopIteration, KeyError, TypeError, ValueError) as e:
        print(f"Error processing odds: {e}")
        result["error_odds"] = str(e)

    try:
        home_attack_strength = home_goals / league_data['average_home_goals'] if league_data['average_home_goals'] > 0 else 1
        home_defense_strength = home_goals_against / league_data['average_away_goals'] if league_data['average_away_goals'] > 0 else 1
        away_attack_strength = away_goals / league_data['average_away_goals'] if league_data['average_away_goals'] > 0 else 1
        away_defense_strength = away_goals_against / league_data['average_home_goals'] if league_data['average_home_goals'] > 0 else 1

        result.update({
            "home_attack_strength": home_attack_strength,
            "home_defense_strength": home_defense_strength,
            "away_attack_strength": away_attack_strength,
            "away_defense_strength": away_defense_strength,
        })
    except Exception as e:
        print(f"Error calculating team strengths: {e}")
        result["error_strengths"] = str(e)

    try:
        lambda_home = normalize_expected_goals(league_data['average_home_goals'] * (home_attack_strength / away_defense_strength) if away_defense_strength > 0 else league_data['average_home_goals'])
        lambda_away = normalize_expected_goals(league_data['average_away_goals'] * (away_attack_strength / home_defense_strength) if home_defense_strength > 0 else league_data['average_away_goals'])

        result.update({
            "expected_goals_home": f"{lambda_home:.2f}",
            "expected_goals_away": f"{lambda_away:.2f}",
        })
    except Exception as e:
        print(f"Error calculating expected goals: {e}")
        result["error_expected_goals"] = str(e)

    try:
        home_form = calculate_form(home_stats.get('form', []))
        away_form = calculate_form(away_stats.get('form', []))

        result.update({
            "home_form": f"{home_form:.2%}",
            "away_form": f"{away_form:.2%}",
        })
    except Exception as e:
        print(f"Error calculating form: {e}")
        result["error_form"] = str(e)

    
        result.update({
            "home_elo": int(home_rating),
            "away_elo": int(home_rating),
        })
    except Exception as e:
        print(f"Error calculating ELO: {e}")
        result["error_elo"] = str(e)

    # Bu kısmı değiştirin
    try:
        features = pd.DataFrame({
            'lambda_home': [lambda_home],
            'lambda_away': [lambda_away],
            'elo_diff': [sigmoid(home_rating - away_rating)],
            'home_form': [home_form],
            'away_form': [away_form]
        })
        labels = pd.Series(league_data['historical_results'])
        enhanced_machine_learning_models = []
        enhanced_predictions = enhanced_machine_learning_models(features, labels)
        home_win_prob, draw_prob, away_win_prob = enhanced_predictions[0]
        result.update({
            "enhanced_model_home_win_prob": f"{home_win_prob:.2%}",
            "enhanced_model_draw_prob": f"{draw_prob:.2%}",
            "enhanced_model_away_win_prob": f"{away_win_prob:.2%}",
        })
    except Exception as e:
        print(f"Error in enhanced machine learning models: {e}")
        result["error_enhanced_ml"] = str(e)

    try:
        home_form_forecast = enhanced_time_series_analysis(league_data['historical_home_form'])
        away_form_forecast = enhanced_time_series_analysis(league_data['historical_away_form'])

        result.update({
            "home_form_forecast": f"{home_form_forecast:.2%}",
            "away_form_forecast": f"{away_form_forecast:.2%}",
        })
    except Exception as e:
        print(f"Error in enhanced time series analysis: {e}")
        result["error_time_series"] = str(e)

    try:
        home_performance_prob = bayes_network_analysis(home_stats)
        away_performance_prob = bayes_network_analysis(away_stats)

        result.update({
            "home_performance_prob": f"{home_performance_prob:.2%}",
            "away_performance_prob": f"{away_performance_prob:.2%}",
        })
    except Exception as e:
        print(f"Error in Bayes network analysis: {e}")
        result["error_bayes_network"] = str(e)

    try:
        dc_prob = dixon_coles_model(home_goals, away_goals, home_attack_strength, away_attack_strength, home_defense_strength, away_defense_strength)
        
        result["dixon_coles_prob"] = f"{dc_prob:.2%}"
    except Exception as e:
        print(f"Error in Dixon-Coles model: {e}")
        result["error_dixon_coles"] = str(e)

    try:
        bayesian_home_win, bayesian_draw, bayesian_away_win = simple_bayesian_analysis(home_attack_strength, away_attack_strength, home_defense_strength, away_defense_strength)

        result.update({
            "bayesian_home_win": f"{bayesian_home_win:.2%}",
            "bayesian_draw": f"{bayesian_draw:.2%}",
            "bayesian_away_win": f"{bayesian_away_win:.2%}",
        })
    except Exception as e:
        print(f"Error in Bayesian analysis: {e}")
        result["error_bayesian"] = str(e)

    try:
        league_level = 1.0
        match_importance = 1.0

        final_home_win_prob, final_draw_prob, final_away_win_prob = calculate_final_probabilities(
            home_rating, away_rating, home_odds, draw_odds, away_odds, home_form, away_form,
            home_performance_prob, away_performance_prob, league_level, match_importance
        )

        result.update({
            "final_home_win_probability": f"{final_home_win_prob:.2%}",
            "final_draw_probability": f"{final_draw_prob:.2%}",
            "final_away_win_probability": f"{final_away_win_prob:.2%}",
        })

        # Değer analizi
        home_value, draw_value, away_value = analyze_odds_value(
            home_odds, draw_odds, away_odds, 
            final_home_win_prob, final_draw_prob, final_away_win_prob
        )

        result.update({
            "home_value": f"{home_value:.2%}",
            "draw_value": f"{draw_value:.2%}",
            "away_value": f"{away_value:.2%}",
        })

    except Exception as e:
        print(f"Error in final probability and value calculation: {e}")
        result.update({
            "final_home_win_probability": "0%",
            "final_draw_probability": "0%",
            "final_away_win_probability": "0%",
            "home_value": "0%",
            "draw_value": "0%",
            "away_value": "0%",
        })

    return result

    try:
        kelly_home = kelly_criterion(final_home_win_prob, home_odds)
        kelly_draw = kelly_criterion(final_draw_prob, draw_odds)
        kelly_away = kelly_criterion(final_away_win_prob, away_odds)

        value_home = calculate_value(final_home_win_prob, home_odds)
        value_draw = calculate_value(final_draw_prob, draw_odds)
        value_away = calculate_value(final_away_win_prob, away_odds)

        result.update({
            "kelly_criterion_home": f"{kelly_home:.2%}",
            "kelly_criterion_draw": f"{kelly_draw:.2%}",
            "kelly_criterion_away": f"{kelly_away:.2%}",
            "value_betting_home": f"{value_home:.2%}",
            "value_betting_draw": f"{value_draw:.2%}",
            "value_betting_away": f"{value_away:.2%}",
        })
    except Exception as e:
        print(f"Error in Kelly criterion and value betting calculation: {e}")
        result["error_kelly_value"] = str(e)

    try:
        simulation_results = simulate_match(
            lambda_home,
            lambda_away,
            home_elo,
            away_elo,
            home_form,
            away_form,
            home_performance_prob,
            away_performance_prob
        )
        result.update({
            "simulated_home_win_prob": f"{simulation_results['home_win_prob']:.2%}",
            "simulated_draw_prob": f"{simulation_results['draw_prob']:.2%}",
            "simulated_away_win_prob": f"{simulation_results['away_win_prob']:.2%}",
            "simulated_avg_total_goals": f"{simulation_results['avg_total_goals']:.2f}",
            "simulated_both_teams_score_prob": f"{simulation_results['both_teams_score_prob']:.2%}",
            "simulated_over_2_5_prob": f"{simulation_results['over_2_5_prob']:.2%}",
            "simulated_home_win_ht_prob": f"{simulation_results['home_win_ht_prob']:.2%}",
            "simulated_draw_ht_prob": f"{simulation_results['draw_ht_prob']:.2%}",
            "simulated_away_win_ht_prob": f"{simulation_results['away_win_ht_prob']:.2%}",
        })
    except Exception as e:
        print(f"Error in match simulation: {e}")
        result["error_simulation"] = str(e)

    return result

def get_league_data(league_id, season):
    return {
        'average_home_goals': 1.5,
        'average_away_goals': 1.2,
        'historical_features': [[1.5, 1.2, 0.5, 0.6, 0.4] for _ in range(100)],
        'historical_results': [0 for _ in range(33)] + [1 for _ in range(34)] + [2 for _ in range(33)],
        'historical_home_form': [0.5 + 0.1 * np.random.randn() for _ in range(100)],
        'historical_away_form': [0.5 + 0.1 * np.random.randn() for _ in range(100)]
    }

def analyze_odds_value(home_odds, draw_odds, away_odds, final_prob_home, final_prob_draw, final_prob_away):
    home_value = (final_prob_home * home_odds) - 1
    draw_value = (final_prob_draw * draw_odds) - 1
    away_value = (final_prob_away * away_odds) - 1
    
    return home_value, draw_value, away_value

def analyze_fixtures(date, start_time, end_time):
    fixtures = get_fixtures(date, start_time, end_time)
    results = []

    for fixture in fixtures:
        odds = get_odds(fixture['fixture']['id'])
        home_stats = get_team_statistics(fixture['teams']['home']['id'], fixture['league']['id'], fixture['league']['season'])
        away_stats = get_team_statistics(fixture['teams']['away']['id'], fixture['league']['id'], fixture['league']['season'])
        h2h = get_h2h(fixture['teams']['home']['id'], fixture['teams']['away']['id'])
        league_data = get_league_data(fixture['league']['id'], fixture['league']['season'])

        result = analyze_match(fixture, odds, home_stats, away_stats, h2h, league_data)
        results.append(result)

    return results

def main():
    # Örnek tarih ve saat aralığı
    example_date = "2024-08-05"
    example_start_time = "10:00"
    example_end_time = "20:00"
    
    while True:
        # Kullanıcıdan tarih ve saat bilgilerini al
        date_input = input(f"Analiz etmek istediginiz tarihi girin (YYYY-MM-DD formatinda, ornek: {example_date}): ")
        if not date_input:
            print("Tarih girişi yapılmadı. Tekrar deneyin.")
            continue

        start_time_input = input(f"Baslangic saatini girin (HH:MM formatinda, ornek: {example_start_time}): ")
        if not start_time_input:
            print("Başlangıç saati girişi yapılmadı. Tekrar deneyin.")
            continue

        end_time_input = input(f"Bitis saatini girin (HH:MM formatinda, ornek: {example_end_time}): ")
        if not end_time_input:
            print("Bitiş saati girişi yapılmadı. Tekrar deneyin.")
            continue

        # Girilen tarihi datetime objesine çevir
        try:
            analysis_date = datetime.strptime(date_input, "%Y-%m-%d").date()
        except ValueError:
            print("Gecersiz tarih formati. Lutfen YYYY-MM-DD formatinda girin.")
            continue

        # Girilen saatleri time objesine çevir
        try:
            start_time = datetime.strptime(start_time_input, "%H:%M").time()
            end_time = datetime.strptime(end_time_input, "%H:%M").time()
        except ValueError:
            print("Gecersiz saat formati. Lutfen HH:MM formatinda girin.")
            continue

        break  # Tüm girdiler doğruysa döngüden çık

    # Analizi çalıştır
    try:
        analysis_results = analyze_fixtures(analysis_date.strftime("%Y-%m-%d"), start_time, end_time)
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return

    if not analysis_results:
        print(f"No matches found on {analysis_date} between {start_time} and {end_time}")
        return

    # Sonuçları yazdır
    for result in analysis_results:
        print(json.dumps(result, indent=2))

    # Sonuçları JSON dosyasına kaydet
    with open(f"analysis_results_{analysis_date}.json", "w") as f:
        json.dump(analysis_results, f, indent=2)
    print(f"Results saved to analysis_results_{analysis_date}.json")

    # Sonuçları veritabanına kaydet
    conn = sqlite3.connect('match_analysis.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS match_analysis
                 (date text, fixture_id integer, home_team text, away_team text, 
                  home_win_prob real, draw_prob real, away_win_prob real)''')
    
    for result in analysis_results:
        home_win_prob = float(result.get('final_home_win_probability', '0%').strip('%')) / 100
        draw_prob = float(result.get('final_draw_probability', '0%').strip('%')) / 100
        away_win_prob = float(result.get('final_away_win_probability', '0%').strip('%')) / 100

        c.execute("INSERT INTO match_analysis VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (str(analysis_date), result['fixture_id'], result['home_team'], result['away_team'],
                   home_win_prob, draw_prob, away_win_prob))
    conn.commit()
    conn.close()
    print("Results saved to database")

    # Sonuçları görselleştir
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(analysis_results)), 
            [float(r['final_home_win_probability'].strip('%')) for r in analysis_results])
    plt.title(f"Home Win Probabilities on {analysis_date}")
    plt.xlabel("Match Index")
    plt.ylabel("Probability (%)")
    plt.savefig(f"home_win_probs_{analysis_date}.png")
    plt.close()
    print(f"Home win probability chart saved as home_win_probs_{analysis_date}.png")

    # Özet istatistikler hesapla
    if analysis_results:
        avg_home_win_prob = sum(float(r['final_home_win_probability'].strip('%')) for r in analysis_results) / len(analysis_results)
        avg_draw_prob = sum(float(r['final_draw_probability'].strip('%')) for r in analysis_results) / len(analysis_results)
        avg_away_win_prob = sum(float(r['final_away_win_probability'].strip('%')) for r in analysis_results) / len(analysis_results)

        print(f"Average probabilities on {analysis_date}:")
        print(f"Home Win: {avg_home_win_prob:.2f}%")
        print(f"Draw: {avg_draw_prob:.2f}%")
        print(f"Away Win: {avg_away_win_prob:.2f}%")

        # Yüksek ev sahibi kazanma olasılığına sahip maçları filtrele
        high_home_win_matches = [r for r in analysis_results if float(r['final_home_win_probability'].strip('%')) > 70]
        print(f"Matches with high home win probability (>70%):")
        for match in high_home_win_matches:
            print(f"{match['home_team']} vs {match['away_team']}: {match['final_home_win_probability']}")

    print("Analysis completed successfully.")
    print(f"Number of matches analyzed: {len(analysis_results)}")

if __name__ == "__main__":
    main()