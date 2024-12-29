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