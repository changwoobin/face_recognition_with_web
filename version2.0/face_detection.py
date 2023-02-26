from simple_websocket_server import WebSocketServer, WebSocket
import base64, cv2
import numpy as np
import warnings
import os
warnings.simplefilter("ignore", DeprecationWarning)

base_dir = './faces/'
class SimpleEcho(WebSocket):
  def handle(self):
    msg = self.data
    name = msg.split(',')[-1]
    msg = msg.replace(','+name, '')

    dir = os.path.join(base_dir, name)

    img = cv2.imdecode(np.fromstring(base64.b64decode(msg.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)

    face_classifier = cv2.CascadeClassifier(\
                        './haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    pic = []
    for files in os.walk('./faces/'+name):
      pic = files[2]
    
    cnt = len(pic)
    if len(faces) == 1 and cnt < 10:
      (x,y,w,h) = faces[0]
      cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
      face = gray[y:y+h, x:x+w]
      face = cv2.resize(face, (200, 200))
      file_name_path = os.path.join(dir,  str(cnt) + '.jpg')
      cv2.imwrite(file_name_path, face)
      self.send_message(str(cnt+1))


  def connected(self):
    print(self.address, 'connected')

  def handle_close(self):
    print(self.address, 'closed')

server = WebSocketServer('localhost', 3000, SimpleEcho)
server.serve_forever()