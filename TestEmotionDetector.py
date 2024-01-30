import cv2
import numpy as np
from keras.models import model_from_json


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
# Charger le modèle et les poids
json_file = open('C:\\Users\\Oumayma Aoun\\Desktop\\ProjetDataMining\\emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("C:\\Users\\Oumayma Aoun\\Desktop\\ProjetDataMining\\emotion_model.h5")
print("Loaded model from disk")

# Charger l'image
image_path = 'C:\\Users\\Oumayma Aoun\\Desktop\\ProjetDataMining\\image4.jfif'
frame = cv2.imread(image_path)

# Conversion en niveaux de gris pour la détection de visage
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Utiliser le classificateur en cascade pour détecter les visages
face_cascade = cv2.CascadeClassifier('C:\\Users\\Oumayma Aoun\\Desktop\\ProjetDataMining\\haarcascade_frontalface_default.xml')
num_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

# Dessiner les rectangles autour des visages détectés et afficher les prédictions d'émotions
for (x, y, w, h) in num_faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    gray_face = gray_frame[y:y + h, x:x + w]
    cropped_img = cv2.resize(gray_face, (48, 48))
    cropped_img = cropped_img / 255.0  # Normalisation des pixels entre 0 et 1
    cropped_img = np.reshape(cropped_img, (1, 48, 48, 1))  # Redimensionner pour le modèle

    # Prédiction des émotions
    emotion_prediction = emotion_model.predict(cropped_img)
    maxindex = int(np.argmax(emotion_prediction))
    emotion_label = emotion_dict[maxindex]
    cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

# Afficher l'image avec les rectangles entourant les visages détectés et les prédictions d'émotions
cv2.imshow('DataMining: Emotion Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()





