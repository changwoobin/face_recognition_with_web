from flask import Flask, render_template, request, redirect
import Face_recognition_web
import os

app = Flask(__name__)

@app.route("/")
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
    userinfo = name+"_"+day

    dir = os.path.join('./faces/', userinfo)

    if not os.path.exists(dir):
        os.mkdir(dir)
    
    return render_template('face_detection.html', userinfo=userinfo)
  else:
    return redirect('/create')
    
@app.route('/check/', methods=['GET','POST'])
def check():
  name = request.form['name']
  day = request.form['day']
  if os.path.isdir('./faces/'+name+"_"+day):
    return render_template('check_account.html', userinfo=name+'_'+day)

  else:
    return redirect('/')

@app.route("/over")
def over():
  Face_recognition_web.trainModel()
  return render_template("after_signup.html")

@app.route('/check_over')
def check_over():
  return render_template("after_signin.html")

if __name__ == "__main__":
  app.run(debug=True)