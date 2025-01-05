import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def create_probability_chart(home_team, away_team, probabilities, model_name):
    """Create bar chart for match outcome probabilities"""
    fig = go.Figure()
    
    outcomes = ['Home Win', 'Draw', 'Away Win']
    colors = ['#2ecc71', '#95a5a6', '#e74c3c']
    
    fig.add_trace(go.Bar(
        x=outcomes,
        y=probabilities,
        text=[f'{p:.1%}' for p in probabilities],
        textposition='auto',
        marker_color=colors
    ))
    
    fig.update_layout(
        title=f"{model_name} Prediction: {home_team} vs {away_team}",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        showlegend=False
    )
    
    return fig

def create_form_chart(home_form, away_form, home_team, away_team):
    """Create radar chart comparing team forms"""
    categories = ['Wins', 'Goals Scored', 'Clean Sheets', 'Form', 'Goal Diff']

    # Normalize form values for radar chart
    home_stats = [
        home_form['wins'] / 5.0,  # Normalize wins
        home_form['avg_goals_scored'] / 3.0,  # Normalize goals
        (5 - home_form['avg_goals_conceded']) / 5.0,  # Clean sheets proxy
        home_form['form_score'],  # Already normalized
        (home_form['avg_goals_scored'] - home_form['avg_goals_conceded'] + 2) / 4.0  # Goal diff normalized
    ]

    away_stats = [
        away_form['wins'] / 5.0,
        away_form['avg_goals_scored'] / 3.0,
        (5 - away_form['avg_goals_conceded']) / 5.0,
        away_form['form_score'],
        (away_form['avg_goals_scored'] - away_form['avg_goals_conceded'] + 2) / 4.0
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=home_stats,
        theta=categories,
        fill='toself',
        name=home_team
    ))

    fig.add_trace(go.Scatterpolar(
        r=away_stats,
        theta=categories,
        fill='toself',
        name=away_team
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Team Form Comparison"
    )

    return fig

def create_history_table(matches_df, home_team, away_team):
    """Create HTML table of historical matches"""
    recent_matches = matches_df[
        ((matches_df['HomeTeam'] == home_team) & (matches_df['AwayTeam'] == away_team)) |
        ((matches_df['HomeTeam'] == away_team) & (matches_df['AwayTeam'] == home_team))
    ].tail(5)
    
    if len(recent_matches) == 0:
        return "No historical matches found"
    
    return recent_matches[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'AwayTeam']].to_html(index=False)

def calculate_combined_prediction(ml_pred, odds_pred, stat_pred):
    """Combine predictions from different models"""
    if odds_pred is None:
        weights = [0.6, 0.0, 0.4]  # ML and Statistical only
        combined = (ml_pred * weights[0] + stat_pred * weights[2])
        total = weights[0] + weights[2]
        return combined / total
    else:
        weights = [0.4, 0.3, 0.3]  # All three models
        combined = (ml_pred * weights[0] + odds_pred * weights[1] + stat_pred * weights[2])
        return combined

def calculate_goal_predictions(home_form, away_form):
    """Calculate goal-related predictions based on team forms"""
    
    # Calculate expected goals using team stats
    home_xg = home_form['avg_goals_scored'] * 1.1  # Home advantage multiplier
    away_xg = away_form['avg_goals_scored'] * 0.9  # Away disadvantage multiplier
    
    total_expected_goals = home_xg + away_xg
    
    # Calculate over probabilities based on expected goals
    over_probs = {
        '0.5': min(0.95, 1 - (0.1 ** total_expected_goals)),
        '1.5': min(0.90, 1 - (0.3 ** total_expected_goals)),
        '2.5': min(0.85, 1 - (0.5 ** total_expected_goals)),
        '3.5': min(0.75, 1 - (0.7 ** total_expected_goals))
    }
    
    # Calculate first half goals (typically 40% of full match goals)
    first_half_goals = total_expected_goals * 0.4
    
    first_half_probs = {
        'IY 0.5': min(0.80, 1 - (0.3 ** first_half_goals)),
        'IY 1.5': min(0.60, 1 - (0.5 ** first_half_goals)),
        'IY 2.5': min(0.30, 1 - (0.7 ** first_half_goals))
    }
    
    # Calculate BTTS (Both Teams To Score) probability
    btts_prob = min(0.90, (home_xg * away_xg) ** 0.5 * 0.7)
    
    # Calculate match tempo based on both teams' scoring and conceding rates
    tempo = min(0.95, (
        (home_form['avg_goals_scored'] + home_form['avg_goals_conceded'] +
         away_form['avg_goals_scored'] + away_form['avg_goals_conceded']) / 8
    ))
    
    # Calculate confidence score based on data reliability
    confidence = min(0.90, (
        (home_form['form_score'] + away_form['form_score']) / 2 +
        (1 - abs(home_xg - away_xg) / (home_xg + away_xg))
    ) * 0.5)
    
    return {
        'total_goals': total_expected_goals,
        'over_probabilities': over_probs,
        'first_half_goals': first_half_goals,
        'first_half_probabilities': first_half_probs,
        'btts_probability': btts_prob,
        'match_tempo': tempo,
        'confidence_score': confidence
    }