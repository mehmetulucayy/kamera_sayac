import cv2
import firebase_admin
from firebase_admin import credentials, db
import time
import threading

# Firebase Ayarları
FIREBASE_CREDENTIALS_PATH = 'react-sayim-firebase-adminsdk-fbsvc-8305e7f44c.json'
# ÖNEMLİ: Bu URL, Firebase hata mesajında belirtilen ve React kodundakiyle aynı olan doğru URL'dir.
FIREBASE_DATABASE_URL = 'https://react-sayim-default-rtdb.europe-west1.firebasedatabase.app/' 

# Firebase başlatma
try:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred, {
        'databaseURL': FIREBASE_DATABASE_URL
    })
    print("Firebase bağlantısı başarılı.")
except Exception as e:
    print(f"Firebase bağlantı hatası: {e}")
    exit()

# HOG (Histogram of Oriented Gradients) kişi algılayıcı
# Bu algılayıcı, insan formlarını tanımak için tasarlanmıştır.
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Kamera başlatma (0 varsayılan web kamerasıdır)
camera = cv2.VideoCapture(0)
# Kamera çözünürlüğünü ayarlama
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Firebase Realtime Database'deki veri yolu
# Bu yol, React uygulamasının veriyi okuduğu yolla eşleşmelidir.
# 'dolmus_user_id' sabit bir ID'dir; eğer her kullanıcı için ayrı veri isterseniz
# burayı dinamik hale getirmeniz gerekir.
firebase_path = 'users/dolmus_user_id/live_count'
ref = db.reference(firebase_path)

# Sayaçlar ve zamanlayıcılar
kare_sayac = 0 # İşlenen kare sayısını takip eder
algilama_araligi = 5 # Her 5 karede bir kişi algılama yapar
kisi_sayisi = 0 # Tespit edilen anlık kişi sayısı
son_gonderme_zamani = time.time() # Son veri gönderme zamanı
gonderme_araligi = 5 # Her 5 saniyede bir Firebase'e veri gönderir

# Firebase'e veri gönderme fonksiyonu
def firebase_gonder(kisi_sayisi_gonderilecek):
    try:
        # Firebase'e doğrudan sayı değerini gönderiyoruz.
        # Bu değer, tespit edilen kişi sayısının 1 fazlası olacaktır.
        ref.set(kisi_sayisi_gonderilecek) 
        print(f"Firebase'e gönderildi: {kisi_sayisi_gonderilecek} kişi")
    except Exception as e:
        print(f"Firebase'e veri gönderme hatası: {e}")

# Ana döngü: Kamera akışını işler
while True:
    # Kameradan bir kare oku
    ret, frame = camera.read()
    if not ret:
        print("Kamera okunamadı. Çıkılıyor...")
        break

    # Her 'algilama_araligi' karede bir kişi algılama yap
    if kare_sayac % algilama_araligi == 0:
        # detectMultiScale: Görüntüdeki nesneleri farklı boyutlarda algılar
        # winStride: Pencere kaydırma adımı
        # padding: Algılama penceresine eklenen dolgu
        # scale: Görüntü küçültme faktörü (1.05 = %5 küçültme)
        rects, weights = hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.05
        )
        kisi_sayisi = len(rects) # Tespit edilen kişi sayısı

    # Tespit edilen her kişi için bir kare çiz ve ID yaz
    for i, (x, y, w, h) in enumerate(rects):
        # Kişi etrafına yeşil bir dikdörtgen çiz
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Kişi ID'sini kare üzerine yaz
        cv2.putText(frame, f"Kişi {i+1}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Ekrana toplam kişi sayısını yaz
    cv2.putText(frame, f"Anlik Kisi Sayisi: {kisi_sayisi}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # İşlenmiş kareyi göster
    cv2.imshow("Kamera", frame)

    kare_sayac += 1

    # Belirli aralıklarla Firebase'e veri gönder
    gecerli_zaman = time.time()
    if gecerli_zaman - son_gonderme_zamani >= gonderme_araligi:
        # Firebase'e gönderilecek kişi sayısını 1 artırıyoruz
        gonderilecek_kisi_sayisi = kisi_sayisi + 1 
        # Veri gönderme işlemini ayrı bir thread'de yap (uygulamanın donmaması için)
        threading.Thread(target=firebase_gonder, args=(gonderilecek_kisi_sayisi,)).start()
        son_gonderme_zamani = gecerli_zaman

    # 'q' tuşuna basılırsa döngüyü kır ve uygulamayı kapat
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak ve pencereleri kapat
camera.release()
cv2.destroyAllWindows()
print("Uygulama kapatıldı.")