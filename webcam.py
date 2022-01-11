# import cv2
# import streamlit as st

# st.title("Webcam Live Feed")
# run = st.checkbox('Run')
# FRAME_WINDOW = st.image([])
# camera = cv2.VideoCapture(0)

# while run:
#     _, frame = camera.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
#     FRAME_WINDOW.image(frame)
# else:
#     st.write('Stopped')

import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import streamlit as st
# from tensorflow.keras.models import load_model

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classLabels = ["mask", "no mask",'improper mask']
# model = load_model("face_Detect.h5")

class VideoProcessor(VideoProcessorBase):

    # @st.cache
    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # magic
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        face = face_cascade.detectMultiScale(gray,1.5,6)
        for (x,y,w,h) in face:
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
            # #roi_gray = gray[y:y+w, x:x+w]
            # roi_col = frame[y:y+h,x:x+w]
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 5)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = gray[y:y + h, x:x + w]
            cv2.putText(gray, "hello world", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2 , cv2.LINE_AA) 

            # crop_img = cv2.resize(roi_color, (224,224))
            # crop_img = np.expand_dims(crop_img, axis=0)
            # prediction = model.predict(crop_img)
            # index = classLabels[np.argmax(prediction)]
            # #print(index)
            # #cv2.putText(, index, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2,color = (0, 0, 255),thickness = 2, cv2.LINE_AA)
            # cv2.putText(gray, index, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2 , cv2.LINE_AA) 

        return gray

webrtc_streamer(key="example", video_processor_factory=VideoProcessor)