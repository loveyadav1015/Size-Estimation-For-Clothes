from fastapi import FastAPI, File, UploadFile, Form
from measurement import extract_features_from_keypoints
from classifier import get_estimated_size

import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MOVENET_MODEL_PATH = BASE_DIR.parent / "models" / "3.tflite"

interpreter = tf.lite.Interpreter(model_path=str(MOVENET_MODEL_PATH))
interpreter.allocate_tensors()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # Reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)
    
    # Setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Make predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    h, w, _ = frame.shape 

    df_features = extract_features_from_keypoints(keypoints_with_scores, h, w)

    if df_features is not None:
        
        predicted_size = get_estimated_size(df_features)
        
        cv2.rectangle(frame, (20, 20), (250, 100), (0, 0, 0), -1)
        
        cv2.putText(frame, f"Size: {predicted_size}", (30, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    else:
        cv2.rectangle(frame, (20, 20), (400, 100), (0, 0, 0), -1)
        
        cv2.putText(frame, f"Stand properly", (30, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    
    cv2.imshow('Size Estiamtion', frame)

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()