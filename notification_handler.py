import os
import asyncio
from typing import Optional, List, Dict
from telegram import Bot
from telegram.error import TelegramError
import streamlit as st

class NotificationHandler:
    def __init__(self):
        self.bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN")
        if not self.bot_token:
            raise ValueError("Telegram bot token bulunamadı!")
        self.bot = Bot(token=self.bot_token)
        self.subscribed_users: Dict[int, Dict] = {}
        
    async def send_message(self, chat_id: int, message: str) -> bool:
        """Send a message to a specific chat"""
        try:
            await self.bot.send_message(chat_id=chat_id, text=message)
            return True
        except TelegramError as e:
            print(f"Mesaj gönderme hatası: {str(e)}")
            return False
            
    def add_subscriber(self, chat_id: int, preferences: Dict) -> None:
        """Add a new subscriber with their notification preferences"""
        self.subscribed_users[chat_id] = preferences
        
    def remove_subscriber(self, chat_id: int) -> None:
        """Remove a subscriber"""
        if chat_id in self.subscribed_users:
            del self.subscribed_users[chat_id]
            
    async def notify_match_event(self, event_type: str, match_data: Dict) -> None:
        """Send match event notifications to subscribed users"""
        message = self._format_match_event(event_type, match_data)
        
        for chat_id, preferences in self.subscribed_users.items():
            if self._should_notify(event_type, preferences):
                await self.send_message(chat_id, message)
                
    def _should_notify(self, event_type: str, preferences: Dict) -> bool:
        """Check if user should be notified based on their preferences"""
        if not preferences.get('enabled', True):
            return False
            
        event_preferences = preferences.get('events', {})
        return event_preferences.get(event_type, True)
        
    def _format_match_event(self, event_type: str, match_data: Dict) -> str:
        """Format match event message"""
        if event_type == 'goal':
            return (f"⚽ GOL!\n"
                   f"{match_data['team']} {match_data['scorer']}\n"
                   f"Dakika: {match_data['minute']}\n"
                   f"Skor: {match_data['score']}")
        elif event_type == 'match_start':
            return (f"🏆 MAÇ BAŞLADI\n"
                   f"{match_data['home_team']} vs {match_data['away_team']}")
        elif event_type == 'match_end':
            return (f"🔚 MAÇ SONA ERDİ\n"
                   f"{match_data['home_team']} {match_data['score']} {match_data['away_team']}")
        elif event_type == 'red_card':
            return (f"🟥 KIRMIZI KART\n"
                   f"{match_data['team']} - {match_data['player']}\n"
                   f"Dakika: {match_data['minute']}")
        else:
            return f"Maç güncellemesi: {str(match_data)}"

# Global notification handler instance
if 'notification_handler' not in st.session_state:
    try:
        st.session_state.notification_handler = NotificationHandler()
    except Exception as e:
        st.error(f"Bildirim sistemi başlatılamadı: {str(e)}")
