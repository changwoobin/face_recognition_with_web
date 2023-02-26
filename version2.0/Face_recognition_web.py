import cv2
import numpy as np
import os, glob

dir_first = os.path.join('./faces/')
base_dir = './faces/'


def trainModel():
    train_data, train_labels = [], []
    dirs = [d for d in glob.glob(base_dir+"/*") if os.path.isdir(d)]
    for dir in dirs:
        id = dir.split('_')[1]          
        files = glob.glob(dir+'/*.jpg')
        for file in files:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            train_data.append(np.asarray(img, dtype=np.uint8))
            train_labels.append(int(id))
    train_data = np.asarray(train_data)
    train_labels = np.int32(train_labels)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(train_data, train_labels)
    model.write('./faces/all_face.xml')




##########################################################




def start_Face_Recognition(frame, userinfo):
    return_Text = ''
    min_accuracy = 80
    username, day = userinfo.split('_')

    face_classifier = cv2.CascadeClassifier(\
                    './haarcascade_frontalface_default.xml')
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(os.path.join(base_dir, 'all_face.xml'))

    dirs = [d for d in glob.glob(base_dir+"/*") if os.path.isdir(d)]
    names = dict([])
    for dir in dirs:
        dir = os.path.basename(dir)
        name, id = dir.split('_')
        names[int(id)] = name

    a = 0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
      face = frame[y:y+h, x:x+w]
      face = cv2.resize(face, (200, 200))
      face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
      label, confidence = model.predict(face)
      if confidence < 400:
          accuracy = int( 100 * (1 -confidence/400))
          if accuracy >= min_accuracy: 
            if label == int(day) and names[label] == username:
              return_Text = "성공"
            else:
              return_Text = "실패"
    cv2.imshow('Face Recognition', frame)

    return return_Text