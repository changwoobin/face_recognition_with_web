from flask import Flask, render_template, Response, request, redirect
import os
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def generate_frames(name, day):
  face_classifier = cv2.CascadeClassifier(\
                        './haarcascade_frontalface_default.xml')
  count = 0
  base_dir = './faces/'

  dir = os.path.join(base_dir, name+'_'+ day)

  if not os.path.exists(dir):
      os.mkdir(dir)

  while True:
    
    ## read the camera frame
    success, frame = camera.read()
    if not success:
      break
    else:
      img = frame.copy()
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      faces = face_classifier.detectMultiScale(gray, 1.3, 5)
      if len(faces) == 1:
        (x,y,w,h) = faces[0]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
        cv2.putText(frame, str(1), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

      ret, buffer = cv2.imencode('.jpg', frame)
      frame = buffer.tobytes()

    yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/create/')
def create_account():
  return render_template('create_account_insert_info.html')

@app.route('/get_info/', methods=['GET', 'POST'])
def get_info():
  if request.method == 'POST':
    name = request.form['name']
    day = request.form['day']  
    return render_template('create_account_take_pic.html', name=name, day=day)
  else:
    return redirect('/create')


@app.route('/video/<string:name>/<string:day>')
def video(name, day):
  return Response(generate_frames(name, day), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)