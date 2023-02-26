from simple_websocket_server import WebSocketServer, WebSocket
import base64, cv2
import numpy as np
import os
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

base_dir = './faces/'
class SimpleEcho(WebSocket):
  def handle(self):
    msg = self.data
    name = msg.split(',')[-1]
    msg = msg.replace(','+name, '')

    dir = os.path.join(base_dir, name)

    
    img = cv2.imdecode(np.fromstring(base64.b64decode(msg.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow('image', img) 
    cv2.waitKey(1)
    

  def connected(self):
    print(self.address, 'connected')

  def handle_close(self):
    print(self.address, 'closed')

server = WebSocketServer('localhost', 3000, SimpleEcho)
server.serve_forever()