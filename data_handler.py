import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class DataHandler:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"
        self.headers = {
            'x-rapidapi-host': "api-football-v1.p.rapidapi.com",
            'x-rapidapi-key': api_key
        }

    def get_live_matches(self) -> Optional[List[Dict]]:
        """Get current live matches"""
        url = f"{self.base_url}/fixtures"
        params = {'live': 'all'}

        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json()['response']
            else:
                print(f"API Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching live matches: {e}")
            return None

    def get_match_statistics(self, fixture_id: int) -> Optional[Dict]:
        """Get real-time statistics for a specific match"""
        url = f"{self.base_url}/fixtures/statistics"
        params = {'fixture': fixture_id}

        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json()['response']
            else:
                print(f"API Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching match statistics: {e}")
            return None

    def get_match_events(self, fixture_id: int) -> Optional[List[Dict]]:
        """Get real-time events for a specific match"""
        url = f"{self.base_url}/fixtures/events"
        params = {'fixture': fixture_id}

        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json()['response']
            else:
                print(f"API Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching match events: {e}")
            return None

    def get_match_lineups(self, fixture_id: int) -> Optional[List[Dict]]:
        """Get team lineups for a specific match"""
        url = f"{self.base_url}/fixtures/lineups"
        params = {'fixture': fixture_id}

        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json()['response']
            else:
                print(f"API Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching match lineups: {e}")
            return None

    def load_historical_data(self, csv_path):
        """Load historical match data from CSV"""
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            score_columns = ['FTHG', 'FTAG', 'HTHG', 'HTAG']
            for col in score_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            odds_columns = [col for col in df.columns if any(x in col.upper() for x in ['ODD', 'PROB', 'RATE'])]
            for col in odds_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.fillna(0)
            return df
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return None

    def get_team_stats(self, team_id, league_id, season):
        """Get team statistics from API"""
        url = f"{self.base_url}/teams/statistics"
        params = {
            "team": team_id,
            "league": league_id,
            "season": season
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json()['response']
            else:
                print(f"API Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching team stats: {e}")
            return None

    def get_h2h_matches(self, team1_id, team2_id):
        """Get head-to-head matches from API"""
        url = f"{self.base_url}/fixtures/headtohead"
        params = {
            "h2h": f"{team1_id}-{team2_id}"
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json()['response']
            else:
                print(f"API Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching H2H matches: {e}")
            return None

    def get_team_form(self, team_id, last_n=5):
        """Get team's recent form from API"""
        url = f"{self.base_url}/teams/statistics"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)

        params = {
            "team": team_id,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d")
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                matches = response.json()['response']['fixtures']['played']['total']
                wins = response.json()['response']['fixtures']['wins']['total']
                draws = response.json()['response']['fixtures']['draws']['total']
                return {
                    'matches': matches,
                    'wins': wins,
                    'draws': draws,
                    'losses': matches - wins - draws
                }
            else:
                print(f"API Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching team form: {e}")
            return None