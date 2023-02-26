from simple_websocket_server import WebSocketServer, WebSocket
import base64, cv2
import numpy as np
import warnings
import Face_recognition_web
import os
warnings.simplefilter("ignore", DeprecationWarning)

base_dir = './faces/'
class SimpleEcho(WebSocket):
  def handle(self):
    msg = self.data
    userinfo = msg.split(',')[-2]
    state_number = msg.split(',')[-1]
    msg = msg.replace(','+userinfo+','+state_number,'')
    img = cv2.imdecode(np.fromstring(base64.b64decode(msg.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)

    if state_number == '1':
      result = Face_recognition_web.start_Face_Recognition(img, userinfo)
      if result == "성공":
        self.send_message("성공")
      elif result == "실패":
        self.send_message("실패")


  def connected(self):
    print(self.address, 'connected')

  def handle_close(self):
    print(self.address, 'closed')

server = WebSocketServer('localhost', 4000, SimpleEcho)
server.serve_forever()