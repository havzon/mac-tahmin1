import streamlit as st
from notification_handler import NotificationHandler

st.set_page_config(page_title="Bildirim Ayarları", layout="wide")

st.title("Bildirim Ayarları")
st.markdown("""
Bu sayfada maç bildirimleri için tercihlerinizi ayarlayabilirsiniz.
Bildirimleri almak için:
1. Telegram'da botumuzu bulun: [TBD - Bot Link]
2. Bota /start komutunu gönderin
3. Size özel kod ile bu sayfada kaydolun
""")

# Notification preferences
st.subheader("Bildirim Tercihleri")

enable_notifications = st.toggle("Bildirimleri Etkinleştir", value=True)

if enable_notifications:
    st.write("Hangi olaylar için bildirim almak istiyorsunuz?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        notify_goals = st.checkbox("Goller", value=True)
        notify_cards = st.checkbox("Kartlar", value=True)
        notify_match_start = st.checkbox("Maç Başlangıcı", value=True)
    
    with col2:
        notify_match_end = st.checkbox("Maç Sonu", value=True)
        notify_injuries = st.checkbox("Sakatlıklar", value=False)
        notify_substitutions = st.checkbox("Oyuncu Değişiklikleri", value=False)

    # Save preferences
    if st.button("Tercihleri Kaydet"):
        preferences = {
            'enabled': enable_notifications,
            'events': {
                'goal': notify_goals,
                'card': notify_cards,
                'match_start': notify_match_start,
                'match_end': notify_match_end,
                'injury': notify_injuries,
                'substitution': notify_substitutions
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
    st.info("Bildirimler devre dışı. Maç güncellemelerini almak için bildirimleri etkinleştirin.")

# Bot Kullanım Kılavuzu
st.markdown("---")
st.subheader("Bot Komutları")
st.markdown("""
- /start - Botu başlat ve kullanım bilgilerini al
- /subscribe - Bildirimlere abone ol
- /unsubscribe - Bildirim aboneliğini iptal et
- /preferences - Bildirim tercihlerini görüntüle
- /help - Yardım menüsünü görüntüle
""")
