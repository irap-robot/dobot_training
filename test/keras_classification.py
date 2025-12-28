import cv2
import numpy as np
import time
from tensorflow import keras
import keras

def predict_single_image(model, img, class_names):
    resized_img = cv2.resize(img, (320, 320))
    print(resized_img.shape)
    img_norm = resized_img / 255.0
    img_norm = np.expand_dims(img_norm, axis=0)

    predictions = model.predict(img_norm)
    predicted_class = np.argmax(predictions)

    cv2.putText(img, f"Predicted: {class_names[predicted_class]}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,50,50), 2)

    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence Scores: {predictions[0]}")

    return img

def crop_object(image) :
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  ret,thresh = cv2.threshold(gray_image,100,255,cv2.THRESH_BINARY_INV)
  countour, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  select_contours = []
  for c in countour:
    area = cv2.contourArea(c)
    if area > 1000:
      select_contours.append(c)

  x,y,w,h = cv2.boundingRect(select_contours[0])
  crop_image = image[y:y+h,x:x+w]

  return crop_image

if __name__ == '__main__' : 
    model_path = 'can_classification_model.keras'
    cap = cv2.VideoCapture(4)
    cls_list = ['cube', 'plain']
    # keras.
    model = keras.models.load_model(model_path)

    while cap.isOpened() : 
        ret, frame = cap.read()
        if not ret : 
            break
        
        cropped_img = crop_object(frame)
        detected_img = predict_single_image(model, cropped_img, cls_list)
        
        e = cv2.waitKey(1) & 0xff
        if e == ord('q') : 
            break
       
        cv2.imshow('detected image', detected_img)
