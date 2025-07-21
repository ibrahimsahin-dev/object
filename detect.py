import cv2
import time
import random
import numpy as np
from ultralytics import YOLO

# --- Servo Motor Kontrolü İçin Placeholder Fonksiyonlar ---
# Bu fonksiyonlar gerçek dünyada Arduino/ESP32 ile seri iletişim kurar
# ve servo motorları kontrol eden komutları gönderir.
# Şu an için sadece simülasyon amaçlı print() kullanacaklar.

def set_horizontal_angle(angle):
    """
    Alt tablayı sağa/sola döndüren servo motoru kontrol eder.
    Gerçekte bu fonksiyon, mikrodenetleyiciye seri porttan komut gönderir.
    """
    print(f"[SERVO CONTROL] Yatay Açı Ayarlandı: {angle:.2f} derece")
    # Gerçek uygulamada: ser.write(f"H{angle}\n".encode())
    time.sleep(0.01) # Servo hareket süresini simüle eder

def set_vertical_angle(angle):
    """
    Üst kafayı yukarı/aşağı döndüren servo motoru kontrol eder.
    Gerçekte bu fonksiyon, mikrodenetleyiciye seri porttan komut gönderir.
    """
    print(f"[SERVO CONTROL] Dikey Açı Ayarlandı: {angle:.2f} derece")
    # Gerçek uygulamada: ser.write(f"V{angle}\n".encode())
    time.sleep(0.01) # Servo hareket süresini simüle eder

def atis():
    """
    Silahın ateş etme/hedefleme komutunu gönderir.
    Bu, bir tetiği çekme veya lazeri açma gibi bir eylem olabilir.
    """
    print("[ATTACK] Hedef Vuruldu!")
    # Gerçek uygulamada: ser.write(b"FIRE\n")
    time.sleep(0.5) # Atış sonrası bekleme

# --- Predefined variables (Mevcut kodunuzdan) ---
confidence_score = 0.5
text_color_b = (0,0,0) # black
text_color_w = (255,255,255) # white
background_color = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX

total_fps = 0
average_fps = 0
num_of_frame = 0
video_frames = []

# Load model
model = YOLO("models/best.pt")
labels = model.names
colors = [[random.randint(0,255) for _ in range(0,3)] for _ in labels]

# Load video (Live Camera)
cap = cv2.VideoCapture(0)

width = int(cap.get(3))
height = int(cap.get(4))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("[INFO].. Width:", width)
print("[INFO].. Height:", height)
print("[INFO].. Total Frames:", total_frames)

# --- Kalibrasyon Değerleri ---
# Bu değerler, piksel farklarını gerçek servo açılarına dönüştürmek için önemlidir.
# Deneme yanılma ile veya daha gelişmiş kalibrasyon yöntemleriyle bulunmalıdır.
# Örneğin, 1 piksel yatayda kaç dereceye denk geliyor?
PIXEL_TO_DEGREE_X = 0.05 # Örnek değer: Her 1 piksel yatay fark 0.05 dereceye denk gelir
PIXEL_TO_DEGREE_Y = 0.04 # Örnek değer: Her 1 piksel dikey fark 0.04 dereceye denk gelir

# Servo başlangıç açıları (genellikle kameranın baktığı merkeze ayarlanır)
current_horizontal_angle = 90.0 # Başlangıç yatay açı (0-180 arası)
current_vertical_angle = 90.0   # Başlangıç dikey açı (0-180 arası)

# Servo açı limitleri
MIN_H_ANGLE = 0
MAX_H_ANGLE = 180
MIN_V_ANGLE = 0
MAX_V_ANGLE = 180

# Kamera merkez noktası (atışın nişangahı)
camera_center_x = width // 2
camera_center_y = height // 2

# --- Hedef Seçimi (İsteğe Bağlı) ---
target_class_name = "red_balloon" # Hangi sınıfı hedeflemek istiyorsunuz? Örneğin 'person'
target_found_in_frame = False

while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Nesne tespiti ve takibi
    results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")[0]
    boxes = np.array(results.boxes.data.tolist())

    target_to_aim = None
    target_distance_to_center = float('inf') # Hedefin merkeze uzaklığı

    for box in boxes:
        if len(box) == 7:
            x1, y1, x2, y2, track_id, score, class_id = box
            x1, y1, x2, y2, class_id, track_id = int(x1), int(y1), int(x2), int(y2), int(class_id), int(track_id)
        else:
            x1, y1, x2, y2, score, class_id = box
            x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)
            track_id = None

        box_color = colors[class_id]
        class_name = results.names[class_id]

        if score > confidence_score:
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            score_percent = score * 100
            if track_id is not None:
                text = f"ID: {track_id} {class_name}: %{score_percent:.2f}"
            else:
                text = f"{class_name}: %{score_percent:.2f}"

            text_loc = (x1, y1-10)
            labelSize, baseLine = cv2.getTextSize(text, font, 1, 1)
            cv2.rectangle(frame,
                          (x1, y1 - 10 - labelSize[1]),
                          (x1 + labelSize[0], int(y1 + baseLine-10)),
                          box_color,
                          cv2.FILLED)
            cv2.putText(frame, text, (x1, y1-10), font, 1, text_color_w, thickness=1)

            # --- Hedef Seçimi ve Vurma Mantığı ---
            if class_name == target_class_name: # Sadece belirli bir sınıfı hedefle
                target_found_in_frame = True
                object_center_x = (x1 + x2) // 2
                object_center_y = (y1 + y2) // 2

                # Hedefin merkeze olan Manhattan mesafesi (yaklaşık)
                distance = abs(object_center_x - camera_center_x) + abs(object_center_y - camera_center_y)

                # Nişangaha en yakın hedefi seç
                if distance < target_distance_to_center:
                    target_distance_to_center = distance
                    target_to_aim = (object_center_x, object_center_y)

                # Hedefin merkezini işaretle
                cv2.circle(frame, (object_center_x, object_center_y), 5, (0, 0, 255), -1)

    # --- Nişangahı Çizme ---
    cv2.circle(frame, (camera_center_x, camera_center_y), 5, (255, 255, 255), -1) # Beyaz nokta nişangah

    # Eğer bir hedef tespit edildiyse, servoları ayarla ve ateş etme emri ver
    if target_to_aim:
        obj_center_x, obj_center_y = target_to_aim

        # Piksel farklarını hesapla
        diff_x = obj_center_x - camera_center_x
        diff_y = obj_center_y - camera_center_y

        # Yeni açıları hesapla
        # Sağa gitmek için yatay açıyı artır, sola gitmek için azalt
        # Aşağı gitmek için dikey açıyı artır (görüntüde y aşağı doğru artar), yukarı gitmek için azalt
        new_horizontal_angle = current_horizontal_angle + (diff_x * PIXEL_TO_DEGREE_X)
        new_vertical_angle = current_vertical_angle + (diff_y * PIXEL_TO_DEGREE_Y)

        # Açıları limitler içinde tut
        new_horizontal_angle = np.clip(new_horizontal_angle, MIN_H_ANGLE, MAX_H_ANGLE)
        new_vertical_angle = np.clip(new_vertical_angle, MIN_V_ANGLE, MAX_V_ANGLE)

        # Servo motorları ayarla
        if abs(diff_x) > 5 or abs(diff_y) > 5: # Küçük hareketleri ignore et (titreşimi azaltır)
            set_horizontal_angle(new_horizontal_angle)
            set_vertical_angle(new_vertical_angle)
            current_horizontal_angle = new_horizontal_angle
            current_vertical_angle = new_vertical_angle
        else: # Hedef merkeze yakınsa, ateş etme emri ver
            print("[INFO] Hedef nişangah içinde. Ateş ediliyor...")
            atis() # Atış fonksiyonunu çağır

        cv2.putText(frame, f"Hedef: {target_class_name}", (10, 80), font, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"H_Angle: {current_horizontal_angle:.1f}", (10, 120), font, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"V_Angle: {current_vertical_angle:.1f}", (10, 160), font, 1, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Hedef Bulunamadi", (10, 80), font, 1, (0, 0, 255), 2)


    end = time.time()
    num_of_frame += 1
    fps = 1 / (end-start)
    total_fps += fps
    average_fps = total_fps / num_of_frame
    avg_fps = float("{:.2f}".format(average_fps))

    cv2.rectangle(frame, (10,2), (280,50), background_color, -1)
    cv2.putText(frame, "FPS: "+str(avg_fps), (20,40), font, 1.5, text_color_b, thickness=3)

    video_frames.append(frame)
    print(f"{num_of_frame} Frames Processed")

    cv2.imshow("Test", frame)
    if cv2.waitKey(20) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("[INFO].. Video is creating.. please wait !")
save_path = "results/test_vid_res.mp4"
writer = cv2.VideoWriter(save_path,
                         cv2.VideoWriter_fourcc(*'XVID'),
                         int(avg_fps),
                         (width,height))

for frame in video_frames:
    writer.write(frame)

writer.release()
print("[INFO].. Video is saved in "+save_path)