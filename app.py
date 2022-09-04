from random import randint
from werkzeug.utils import secure_filename
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from flask import Flask, render_template, request
import cv2
from PIL import Image
from keras.models import load_model 
from tensorflow.keras.utils import load_img, img_to_array
from keras.preprocessing import image
import numpy as np
import uuid

app = Flask(__name__)

model = load_model("model/pred.h5")

testing_folder = "static/testing/"
image_folder = "static/uploads"
app.config["UPLOAD_FOLDER"] = image_folder
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def get_random_image(folder, labels):
    data = []
    for label in labels:
        path = os.path.join(folder, label)
        x = os.listdir(path)
        select = randint(0, 73)
        data.append(path + "/" + x[select])
    return data, labels

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', data=get_random_image(testing_folder, labels))

# model classes
labels = ["glioma_tumor", "no_tumor", "meningioma_tumor", "pituitary_tumor"]

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST'])
def img_pred():
    imagefile = request.files['imagefile']
    if imagefile and allowed_file(imagefile.filename):
        filename = str(uuid.uuid4().hex) + secure_filename(imagefile.filename)
        # myfilename = secure_filename(imagefile.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        imagefile.save(image_path)
    else:
        return render_template('index.html', pred_error="Please upload a valid image file", data=get_random_image(testing_folder, labels))
    img = load_img(image_path)
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage, (150, 150))
    img = img.reshape(1, 150, 150, 3)
    p = model.predict(img)
    p = np.argmax(p, axis=1)[0]
    return render_template('index.html', prediction=labels[p], user_image=image_path, data=get_random_image(testing_folder, labels))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
