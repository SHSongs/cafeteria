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

@app.route('/upload-processing-test', methods=['post'])
def upload_processing_test():
    cnt = str(request.data)
    # my_json = json.loads(cnt)
    # print(my_json)
    return f"how many people : {cnt}\n remain seats : "

if __name__ == '__main__':
    app.run(host='0.0.0.0')