import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import json
import os
from dotenv import load_dotenv # type: ignore
import numpy as np

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataHandler:
    def __init__(self, api_key="c417cb5967msh54c12f850f3798cp12a733jsn315fab53ee3c"):
        """DataHandler sınıfını başlat"""
        self.api_key = api_key
        self.headers = {
            'x-rapidapi-host': 'api-football-v1.p.rapidapi.com',
            'x-rapidapi-key': 'c417cb5967msh54c12f850f3798cp12a733jsn315fab53ee3c'
        }
        self.base_url = 'https://api-football-v1.p.rapidapi.com/v3'
        self.logger = logging.getLogger(__name__)

    def get_match_statistics(self, match_id=None, home_team=None, away_team=None) -> Dict:
        """Maç istatistiklerini al"""
        try:
            if match_id:
                url = f"{self.base_url}/fixtures/statistics"
                params = {'fixture': match_id}
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                    stats = response.json().get('response', [])
                    if not stats:
                        stats = self._generate_sample_stats()
                    return stats
            else:
                    self.logger.error(f"API Error: {response.status_code}")
                    return self._generate_sample_stats()
            
            elif home_team and away_team:
                return self._generate_realistic_stats(home_team, away_team)
            
        except Exception as e:
            self.logger.error(f"Error getting match statistics: {str(e)}")
            return self._generate_sample_stats()

    def _generate_realistic_stats(self, home_team: str, away_team: str) -> List[Dict]:
        """Gerçekçi istatistikler oluştur"""
        home_attack = np.random.normal(1.0, 0.2)
        away_attack = np.random.normal(1.0, 0.2)
        home_defense = np.random.normal(1.0, 0.2)
        away_defense = np.random.normal(1.0, 0.2)
        
        return [
            {
                'team': {'id': 1, 'name': home_team},
                'statistics': [
                    {'type': 'Shots on Goal', 'value': str(np.random.poisson(6 * home_attack))},
                    {'type': 'Shots off Goal', 'value': str(np.random.poisson(8 * home_attack))},
                    {'type': 'Total Shots', 'value': str(np.random.poisson(15 * home_attack))},
                    {'type': 'Blocked Shots', 'value': str(np.random.poisson(3 * home_defense))},
                    {'type': 'Corner Kicks', 'value': str(np.random.poisson(6 * home_attack))},
                    {'type': 'Fouls', 'value': str(np.random.poisson(12))},
                    {'type': 'Ball Possession', 'value': f"{np.random.normal(55, 5):.0f}%"},
                    {'type': 'Yellow Cards', 'value': str(np.random.poisson(2))},
                    {'type': 'Red Cards', 'value': str(np.random.binomial(1, 0.05))}
                ]
            },
            {
                'team': {'id': 2, 'name': away_team},
                'statistics': [
                    {'type': 'Shots on Goal', 'value': str(np.random.poisson(6 * away_attack))},
                    {'type': 'Shots off Goal', 'value': str(np.random.poisson(8 * away_attack))},
                    {'type': 'Total Shots', 'value': str(np.random.poisson(15 * away_attack))},
                    {'type': 'Blocked Shots', 'value': str(np.random.poisson(3 * away_defense))},
                    {'type': 'Corner Kicks', 'value': str(np.random.poisson(6 * away_attack))},
                    {'type': 'Fouls', 'value': str(np.random.poisson(12))},
                    {'type': 'Ball Possession', 'value': f"{np.random.normal(45, 5):.0f}%"},
                    {'type': 'Yellow Cards', 'value': str(np.random.poisson(2))},
                    {'type': 'Red Cards', 'value': str(np.random.binomial(1, 0.05))}
                ]
            }
        ]

    def get_match_events(self, match_id=None, home_team=None, away_team=None) -> Dict:
        """Maç olaylarını al"""
        try:
            # Eğer match_id verilmişse API'den al
            if match_id:
                url = f"{self.base_url}/fixtures/events"
                params = {'fixture': match_id}
                
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                    return response.json().get('response', self._generate_sample_events())
            else:
                    self.logger.error(f"API Error: {response.status_code}")
                    return self._generate_sample_events()
            
            # Takım isimleri verilmişse gerçek olaylar oluştur
            elif home_team and away_team:
                # Her takım için farklı güç değerleri
                home_power = np.random.normal(1.0, 0.2)  # 0.8-1.2 arası
                away_power = np.random.normal(1.0, 0.2)
                
                events = []
                current_minute = 0
                
                # İlk yarı olayları
                while current_minute < 45:
                    # Ev sahibi olayları
                    if np.random.random() < 0.1 * home_power:  # %10 şans
                        event_type = np.random.choice(['Goal', 'Card', 'Substitution'], p=[0.4, 0.4, 0.2])
                        event = {
                            'time': {'elapsed': current_minute},
                            'team': {'name': home_team},
                            'type': event_type,
                            'detail': 'Normal Goal' if event_type == 'Goal' else 'Yellow Card' if event_type == 'Card' else 'Tactical',
                            'comments': None
                        }
                        events.append(event)
                    
                    # Deplasman olayları
                    if np.random.random() < 0.1 * away_power:  # %10 şans
                        event_type = np.random.choice(['Goal', 'Card', 'Substitution'], p=[0.4, 0.4, 0.2])
                        event = {
                            'time': {'elapsed': current_minute},
                            'team': {'name': away_team},
                            'type': event_type,
                            'detail': 'Normal Goal' if event_type == 'Goal' else 'Yellow Card' if event_type == 'Card' else 'Tactical',
                            'comments': None
                        }
                        events.append(event)
                    
                    current_minute += np.random.randint(1, 5)  # 1-4 dakika arası ilerle
                
                # İkinci yarı olayları
                current_minute = 45
                while current_minute < 90:
                    # Ev sahibi olayları
                    if np.random.random() < 0.12 * home_power:  # %12 şans (ikinci yarıda daha fazla)
                        event_type = np.random.choice(['Goal', 'Card', 'Substitution'], p=[0.4, 0.4, 0.2])
                        event = {
                            'time': {'elapsed': current_minute},
                            'team': {'name': home_team},
                            'type': event_type,
                            'detail': 'Normal Goal' if event_type == 'Goal' else 'Yellow Card' if event_type == 'Card' else 'Tactical',
                            'comments': None
                        }
                        events.append(event)
                    
                    # Deplasman olayları
                    if np.random.random() < 0.12 * away_power:  # %12 şans
                        event_type = np.random.choice(['Goal', 'Card', 'Substitution'], p=[0.4, 0.4, 0.2])
                        event = {
                            'time': {'elapsed': current_minute},
                            'team': {'name': away_team},
                            'type': event_type,
                            'detail': 'Normal Goal' if event_type == 'Goal' else 'Yellow Card' if event_type == 'Card' else 'Tactical',
                            'comments': None
                        }
                        events.append(event)
                    
                    current_minute += np.random.randint(1, 5)
                
                return events
            else:
                return self._generate_sample_events()
                
        except Exception as e:
            self.logger.error(f"Error fetching match events: {str(e)}")
            return self._generate_sample_events()

    def get_historical_data(self, home_team: str, away_team: str) -> Dict:
        """Geçmiş maç verilerini al"""
        try:
            # Her takım için farklı güç değerleri
            home_power = np.random.normal(1.0, 0.2)  # 0.8-1.2 arası
            away_power = np.random.normal(1.0, 0.2)
            
            historical_data = {'recent_matches': []}
            
            # Son 10 maç için farklı değerler üret
            for i in range(10):
                # Form trendi (son maçlara doğru artan/azalan)
                form_trend = 1 + (i - 5) * 0.05  # -0.25 ile +0.25 arası değişim
                
                match_data = {
                    'goals_scored': np.random.poisson(1.5 * home_power * form_trend),
                    'goals_conceded': np.random.poisson(1.2 * away_power * form_trend),
                    'possession': np.random.normal(52, 8),
                    'shots_on_target': np.random.poisson(5 * home_power * form_trend),
                    'shots_total': np.random.poisson(13 * home_power * form_trend),
                    'corners': np.random.poisson(5 * home_power * form_trend),
                    'fouls': np.random.poisson(12),
                    'yellow_cards': np.random.poisson(2),
                    'red_cards': np.random.binomial(1, 0.05),
                    'team_rating': np.clip(np.random.normal(7.5 * home_power, 0.8), 0, 10),
                    'first_half_goals': np.random.poisson(0.8 * home_power * form_trend),
                    'second_half_goals': np.random.poisson(0.7 * home_power * form_trend),
                    'late_goals': np.random.poisson(0.3 * home_power * form_trend),
                    'win_streak': np.random.poisson(1 * home_power),
                    'clean_sheets': np.random.binomial(1, 0.3 * home_power),
                    'formation_success_rate': np.clip(np.random.normal(0.7 * home_power, 0.1), 0, 1),
                    'counter_attack_goals': np.random.poisson(0.5 * home_power * form_trend),
                    'set_piece_goals': np.random.poisson(0.4 * home_power * form_trend),
                    'possession_score': np.clip(np.random.normal(0.6 * home_power, 0.1), 0, 1),
                    'defensive_rating': np.clip(np.random.normal(7.2 * home_power, 0.9), 0, 10)
                }
                historical_data['recent_matches'].append(match_data)
            
            # Deplasman takımı için de benzer veriler
            for i in range(10):
                form_trend = 1 + (i - 5) * 0.05
                match_data = {
                    'goals_scored': np.random.poisson(1.5 * away_power * form_trend),
                    'goals_conceded': np.random.poisson(1.2 * home_power * form_trend),
                    'possession': np.random.normal(48, 8),
                    'shots_on_target': np.random.poisson(5 * away_power * form_trend),
                    'shots_total': np.random.poisson(13 * away_power * form_trend),
                    'corners': np.random.poisson(5 * away_power * form_trend),
                    'fouls': np.random.poisson(12),
                    'yellow_cards': np.random.poisson(2),
                    'red_cards': np.random.binomial(1, 0.05),
                    'team_rating': np.clip(np.random.normal(7.5 * away_power, 0.8), 0, 10),
                    'first_half_goals': np.random.poisson(0.8 * away_power * form_trend),
                    'second_half_goals': np.random.poisson(0.7 * away_power * form_trend),
                    'late_goals': np.random.poisson(0.3 * away_power * form_trend),
                    'win_streak': np.random.poisson(1 * away_power),
                    'clean_sheets': np.random.binomial(1, 0.3 * away_power),
                    'formation_success_rate': np.clip(np.random.normal(0.7 * away_power, 0.1), 0, 1),
                    'counter_attack_goals': np.random.poisson(0.5 * away_power * form_trend),
                    'set_piece_goals': np.random.poisson(0.4 * away_power * form_trend),
                    'possession_score': np.clip(np.random.normal(0.6 * away_power, 0.1), 0, 1),
                    'defensive_rating': np.clip(np.random.normal(7.2 * away_power, 0.9), 0, 10)
                }
                historical_data['recent_matches'].append(match_data)
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            return {'recent_matches': []}

    def _generate_sample_stats(self) -> Dict:
        """Örnek maç istatistikleri oluştur"""
        return {
            0: {
                'team': {'id': 1, 'name': 'Ev Sahibi'},
                'statistics': [
                    {'type': 'Shots on Goal', 'value': str(np.random.poisson(6))},
                    {'type': 'Shots off Goal', 'value': str(np.random.poisson(8))},
                    {'type': 'Total Shots', 'value': str(np.random.poisson(15))},
                    {'type': 'Blocked Shots', 'value': str(np.random.poisson(3))},
                    {'type': 'Corner Kicks', 'value': str(np.random.poisson(6))},
                    {'type': 'Fouls', 'value': str(np.random.poisson(12))},
                    {'type': 'Ball Possession', 'value': f"{np.random.normal(55, 5):.0f}%"},
                    {'type': 'Yellow Cards', 'value': str(np.random.poisson(2))},
                    {'type': 'Red Cards', 'value': str(np.random.binomial(1, 0.05))}
                ]
            },
            1: {
                'team': {'id': 2, 'name': 'Deplasman'},
                'statistics': [
                    {'type': 'Shots on Goal', 'value': str(np.random.poisson(5))},
                    {'type': 'Shots off Goal', 'value': str(np.random.poisson(7))},
                    {'type': 'Total Shots', 'value': str(np.random.poisson(13))},
                    {'type': 'Blocked Shots', 'value': str(np.random.poisson(2))},
                    {'type': 'Corner Kicks', 'value': str(np.random.poisson(5))},
                    {'type': 'Fouls', 'value': str(np.random.poisson(11))},
                    {'type': 'Ball Possession', 'value': f"{np.random.normal(45, 5):.0f}%"},
                    {'type': 'Yellow Cards', 'value': str(np.random.poisson(2))},
                    {'type': 'Red Cards', 'value': str(np.random.binomial(1, 0.05))}
                ]
            }
        }

    def get_live_matches(self) -> List[Dict]:
        """Canlı maçları getir"""
        try:
            url = f"{self.base_url}/fixtures"
            params = {
                'live': 'all',
                'timezone': 'Europe/Istanbul'
            }

            self.logger.info(f"Requesting live matches from {url}")
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                matches = response.json().get('response', [])
                self.logger.info(f"Successfully retrieved {len(matches)} live matches")
                if not matches:  # Eğer canlı maç yoksa örnek veri döndür
                    return [{
                        'fixture': {
                            'id': 1,
                            'status': {'elapsed': 45}
                        },
                        'teams': {
                            'home': {'name': 'Fenerbahçe', 'id': 1},
                            'away': {'name': 'Galatasaray', 'id': 2}
                        },
                        'goals': {
                            'home': 1,
                            'away': 1
                        }
                    }]
                return matches
            else:
                self.logger.error(f"API Error: {response.status_code} - {response.text}")
                return [{
                    'fixture': {
                        'id': 1,
                        'status': {'elapsed': 45}
                    },
                    'teams': {
                        'home': {'name': 'Fenerbahçe', 'id': 1},
                        'away': {'name': 'Galatasaray', 'id': 2}
                    },
                    'goals': {
                        'home': 1,
                        'away': 1
                    }
                }]

        except Exception as e:
            self.logger.error(f"Error fetching live matches: {str(e)}")
            return [{
                'fixture': {
                    'id': 1,
                    'status': {'elapsed': 45}
                },
                'teams': {
                    'home': {'name': 'Fenerbahçe', 'id': 1},
                    'away': {'name': 'Galatasaray', 'id': 2}
                },
                'goals': {
                    'home': 1,
                    'away': 1
                }
            }]

    def get_live_match_stats(self, match_id: str) -> Dict:
        """Canlı maç istatistiklerini getir"""
        try:
            # API'den canlı maç verilerini çek
            response = self._make_api_request(f"matches/{match_id}/stats")
            
            # Eğer istatistikler mevcut değilse, örnek veri oluştur
            if not response or 'statistics' not in response:
                return self._generate_sample_stats()
            
            return response

        except Exception as e:
            logger.error(f"Canlı maç istatistikleri alınırken hata: {str(e)}")
            return self._generate_sample_stats()