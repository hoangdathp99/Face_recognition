import cv2
import numpy as np # Thư viện đại số tuyến tính cho Python
from keras.preprocessing.image import img_to_array
from keras.models import load_model


face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('files/emotion_model.hdf5', compile=False)
EMOTIONS = ["Tuc gian","Kinh tom","So hai", "Hanh phuc", "Buon ba", "Bat ngo", "Binh thuong"]

camera = cv2.VideoCapture(0)

while True:
    # Chụp hình ảnh từ camera
    ret, frame = camera.read()
    
    # Chuyển từ ảnh màu sang ảnh xám bằng OpenCV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện khuôn mặt trong khung hình
    faces = face_detection.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30,30))
    
    # Tạo khung hình hiển thị tỉ lệ giữa các cảm xúc
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    
    # Chỉ thực hiện nhận biết cảm xúc khi phát hiện được có khuôn mặt trong hình
    if len(faces) > 0:
        # Chỉ thực hiện với khuôn mặt chính trong hình (khuôn mặt có diện tích lớn nhất)
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face
        # Tách phần khuôn mặt vừa tìm được và resize về kích thước 48x48 để chuẩn bị đưa vào bộ mạng Neural Network
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # Thực hiện dự đoán cảm xúc
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        
        # Gán nhãn cảm xúc dự đoán được lên hình
        cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
 
        # In các mức độ của cảm xúc (theo %) lên cửa sổ thứ 2
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)    
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    # Mở 2 cửa sổ
    ## Cửa sổ hiện thị hình ảnh chụp từ camera
    ## Cửa sổ hiện thị mức độ của các cảm xúc (theo %)
    cv2.imshow('Emotion Recognition', frame)
    cv2.imshow("Probabilities", canvas)
    
    # Nhấn phím "q" để kết thúc chương trình
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp chương trình, giải phóng bộ nhớ và camera
camera.release()
cv2.destroyAllWindows()