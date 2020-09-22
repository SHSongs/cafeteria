import os
import json

from flask import Flask, render_template
from flask import send_file, request, Response

import mocking_data

app = Flask(__name__)

@app.route('/')
def hello():
    counter = mocking_data.Generator()
    return str(counter.get_counter())

@app.route('/image')
def image():
    filename = "C:/cafeteria/server/images/cafeteria.png"
    return send_file(filename, mimetype='image/png')

@app.route('/upload', methods=['post', 'get'])
def upload():
    return render_template('upload.html')

@app.route('/upload-processing', methods=['post'])
def upload_processing():
    f = request.files['file']
    f_name = os.path.join('C:/cafeteria/server/images', f.filename)
    try:
        f.save(f_name)
    except Exception:
        print(Exception)
    return send_file(f_name, mimetype="image/png")

seat = 104

@app.route('/upload-processing-test', methods=['post'])
def upload_processing_test():
    global seat
    post = request.data
    my_json = json.loads(post)
    print(my_json)

    if my_json['entrance'] == 'in':
        seat = seat - 1
    elif my_json['entrance'] == 'out':
        seat = seat + 1
    
    file = open("seat.txt", 'w')
    file.write(str(seat))
    file.close()
    return f"json 정보 사람수, 입구출구 여부 : {my_json}\n잔여 좌석 : {seat}"

current = 0

@app.route('/capture', methods=['post'])
def capture():
    up_current()
    return "done"

@app.route('/cafeteria_state', methods=['get', 'post'])
def cafeteria_state():
    return render_template('cafeteria_state.html')

@app.route('/img', methods=['get', 'post'])
def img():
    filename = "C:/cafeteria/server/images/"+"img"+str(get_current())+".jpg"
    return send_file(filename, mimetype='image/png')

def up_current():
    global current
    current += 1

def get_current():
    global current
    return current

if __name__ == '__main__':
    app.run(host='0.0.0.0')