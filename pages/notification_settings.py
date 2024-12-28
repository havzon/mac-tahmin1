# En başta set_page_config - tüm importlardan önce olmalı
import streamlit as st
st.set_page_config(page_title="Bildirim Ayarları", layout="wide")

from notification_handler import NotificationHandler

st.title("Bildirim Ayarları")
st.markdown("""
Bu sayfada gol bildirimleri için tercihlerinizi ayarlayabilirsiniz.
Bildirimleri almak için:
1. Telegram'da botumuzu bulun: [TBD - Bot Link]
2. Bota /start komutunu gönderin
3. Size özel kod ile bu sayfada kaydolun
""")

# Notification preferences
st.subheader("Bildirim Tercihleri")

enable_notifications = st.toggle("Gol Bildirimlerini Etkinleştir", value=True)

if enable_notifications:
    st.write("Hangi durumlarda gol bildirimi almak istiyorsunuz?")

    notify_all_goals = st.checkbox("Tüm Goller", value=True)
    notify_home_goals = st.checkbox("Sadece Ev Sahibi Golleri", value=False)
    notify_away_goals = st.checkbox("Sadece Deplasman Golleri", value=False)

    # Save preferences
    if st.button("Tercihleri Kaydet"):
        preferences = {
            'enabled': enable_notifications,
            'events': {
                'all_goals': notify_all_goals,
                'home_goals': notify_home_goals,
                'away_goals': notify_away_goals
            }
        }

        try:
            # Kullanıcı chat_id'si normalde bot ile etkileşimden gelecek
            # Şimdilik test için sabit bir değer kullanıyoruz
            test_chat_id = 123456  # Bu değer gerçek uygulamada dinamik olacak
            st.session_state.notification_handler.add_subscriber(test_chat_id, preferences)
            st.success("Bildirim tercihleri kaydedildi!")
        except Exception as e:
            st.error(f"Tercihler kaydedilirken hata oluştu: {str(e)}")

else:
    st.info("Bildirimler devre dışı. Gol bildirimlerini almak için bildirimleri etkinleştirin.")

# Bot Kullanım Kılavuzu
st.markdown("---")
st.subheader("Bot Komutları")
st.markdown("""
- /start - Botu başlat ve kullanım bilgilerini al
- /subscribe - Gol bildirimlerine abone ol
- /unsubscribe - Bildirim aboneliğini iptal et
- /preferences - Bildirim tercihlerini görüntüle
- /help - Yardım menüsünü görüntüle
""")