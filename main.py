from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import requests
import base64
import json

FLASK_SERVER = "http://YOUR_FLASK_SERVER_IP:5000"
# FLASK_SERVER = "http://192.168.1.100:5000"


class CamApp(App):
    def build(self):
        self.img1 = Image()
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 3.0)
        return self.img1

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Show frame
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img1.texture = texture

            # Send to server
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            response = requests.post(f"{FLASK_SERVER}/api/search_frame", json={'image': f"data:image/jpeg;base64,{jpg_as_text}"})
            if response.ok:
                result = response.json()
                if result.get('match'):
                    print("✅ Match found:", result['person'])
                else:
                    print("❌ No match found")

CamApp().run()
