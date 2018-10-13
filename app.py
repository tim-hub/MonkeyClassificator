from flask import Flask, redirect, request, flash, request, url_for, render_template
from werkzeug.utils import secure_filename



import keras
import keras.optimizers
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'gif'])

# global model_t
# global model_c

@app.route('/prepare')
def warm_up():
    prepare_transfer_model()
    # prepare_cnn_model()
    return redirect('/')


@app.route('/', methods=['GET', 'POST'])
def predict_by():
    labels = _extract_labels()
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # file.save(os.path.join(UPLOAD_FOLDER, filename))
            # return redirect(url_for('uploaded_file',
            #                         filename=filename))

            result = None
            if (model_t != None):
                result = predict(model_t, file, 300, 300)



            return render_template('index.html', result=result, labels=labels)

    return render_template('index.html', labels=labels)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init():
    print(os.listdir("../"))


def prepare_transfer_model():
    global model_t
    model_t= keras.models.load_model('../incept_adv.h5')

    model_t.compile(loss="categorical_crossentropy",
                  optimizer='adam',
                  metrics=["accuracy"])
    model_t._make_predict_function()

def prepare_cnn_model():
    global model_c
    model_c = keras.models.load_model('../model.h5f')
    model_c.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(lr=1e-3, decay=1e-4),
                      metrics=['accuracy'])
    model_c._make_predict_function()

def _extract_labels():
    label_path = '../input/10-monkey-species/monkey_labels.txt'
    data = pd.read_csv(label_path, sep=',', header=0, index_col=False).rename(columns=lambda x: x.strip())
    labels = data['Common Name']
    return labels


def predict(model, input_img_path, img_width, img_height):

    img = Image.open(input_img_path)
    resized_img = img.resize((img_width, img_height), Image.BICUBIC)
    img_array = np.array(resized_img, dtype='int')
    (img_row, img_column, img_channels) = img_array.shape
    input_img = img_array.reshape(1, img_row, img_column, img_channels)
    classes = model.predict_classes(input_img)

    index = classes[0]
    labels = _extract_labels()
    print((labels[index]))
    return (labels[index])

# prepare_transfer_model()
# predict(model_t, '/Users/tim/Development/PythonProjects/monkeys/input/10-monkey-species/validation/n7/n710.jpg', 300,300)


if __name__ == '__main__':
    app.run()
