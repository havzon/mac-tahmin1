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
        """Add a new subscriber with their goal notification preferences"""
        self.subscribed_users[chat_id] = preferences

    def remove_subscriber(self, chat_id: int) -> None:
        """Remove a subscriber"""
        if chat_id in self.subscribed_users:
            del self.subscribed_users[chat_id]

    async def notify_goal(self, goal_data: Dict) -> None:
        """Send goal notifications to subscribed users based on their preferences"""
        message = self._format_goal_message(goal_data)

        for chat_id, preferences in self.subscribed_users.items():
            if self._should_notify_goal(goal_data, preferences):
                await self.send_message(chat_id, message)

    def _should_notify_goal(self, goal_data: Dict, preferences: Dict) -> bool:
        """Check if user should be notified about this goal based on their preferences"""
        if not preferences.get('enabled', True):
            return False

        events = preferences.get('events', {})

        # Eğer tüm goller için bildirim isteniyorsa
        if events.get('all_goals', True):
            return True

        # Sadece ev sahibi golleri için bildirim isteniyorsa
        if events.get('home_goals', False) and goal_data.get('is_home_team', False):
            return True

        # Sadece deplasman golleri için bildirim isteniyorsa
        if events.get('away_goals', False) and not goal_data.get('is_home_team', False):
            return True

        return False

    def _format_goal_message(self, goal_data: Dict) -> str:
        """Format goal event message"""
        return (f"⚽ GOL!\n"
                f"{goal_data['team']} {goal_data['scorer']}\n"
                f"Dakika: {goal_data['minute']}\n"
                f"Skor: {goal_data['score']}")

# Global notification handler instance
if 'notification_handler' not in st.session_state:
    try:
        st.session_state.notification_handler = NotificationHandler()
    except Exception as e:
        st.error(f"Bildirim sistemi başlatılamadı: {str(e)}")