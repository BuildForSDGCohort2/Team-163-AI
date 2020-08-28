import io
import torch
import torch.nn as nn
import flask
from flask import request, jsonify, send_file
from torchvision import models, transforms
from PIL import Image
from flask_cors import CORS
import os

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app)


@app.route('/', methods=['GET'])
def home():
    try: 
        return send_file(os.path.join('api_info.html'), 'text/html')
    except Exception as e:
        response = flask.jsonify({'error': 'A server error occured, while attempting to render page'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500


@app.route('/api/v1/train', methods=['GET'])
def api_train():
    content = {
        'error': 'not yet available'
    }
    return content, 501


model_roadtype = models.resnet18(pretrained=True)
num_ftrs = model_roadtype.fc.in_features
model_roadtype.fc = nn.Linear(num_ftrs, 3)

model_roadtype.load_state_dict(torch.load('trained_models/model_dict_road_type_classifier.pt'))
model_roadtype.eval()

# asphalt condition model
model_asphalt_condition = models.resnet18(pretrained=True)
num_ftrs = model_asphalt_condition.fc.in_features
model_asphalt_condition.fc = nn.Linear(num_ftrs, 3)

model_asphalt_condition.load_state_dict(torch.load('trained_models/model_dict_asphalt_condition_classifier.pt'))
model_asphalt_condition.eval()

# unpaved condition model
model_unpaved_condition = models.resnet18(pretrained=True)
num_ftrs = model_unpaved_condition.fc.in_features
model_unpaved_condition.fc = nn.Linear(num_ftrs, 2)

model_unpaved_condition.load_state_dict(torch.load('trained_models/model_dict_unpaved_condition_classifier.pt'))
model_unpaved_condition.eval()

# paved condition model
model_paved_condition = models.resnet18(pretrained=True)
num_ftrs = model_paved_condition.fc.in_features
model_paved_condition.fc = nn.Linear(num_ftrs, 3)

model_paved_condition.load_state_dict(torch.load('trained_models/model_dict_paved_condition_classifier.pt'))
model_paved_condition.eval()

trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

road_types = {'0': 'asphalt', '1': 'paved', '2': 'dirt'}
road_conditions = {'0': 'bad', '1': 'fair', '2': 'good'}


@app.route('/api/v1/read', methods=['POST'])
def api_read():
    predicted_condition_idx = 9
    type_confidance = 0
    file = request.files['file']
    image_extensions = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
    if file.filename.split('.')[-1] not in image_extensions:
        response = flask.jsonify({'error': 'Only jpg, jpeg and png are supported'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 415

    image_bytes = file.read()
    road_image = Image.open(io.BytesIO(image_bytes))
    tensor = trans(road_image).unsqueeze(0)
    outputs = model_roadtype.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_type_idx = str(y_hat.item())

    if predicted_type_idx == '0':
        outputs = model_asphalt_condition.forward(tensor)
        _, y_hat = outputs.max(1)
        predicted_condition_idx = str(y_hat.item())

    if predicted_type_idx == '1':
        outputs = model_paved_condition.forward(tensor)
        _, y_hat = outputs.max(1)
        predicted_condition_idx = str(y_hat.item())

    if predicted_type_idx == '2':
        outputs = model_unpaved_condition.forward(tensor)
        _, y_hat = outputs.max(1)
        predicted_condition_idx = str(y_hat.item())

    response = flask.jsonify({
        'type': {
            'prediction': road_types[predicted_type_idx],
            'confidence': type_confidance
        },
        'condition': {
            'prediction': road_conditions[predicted_condition_idx],
            'confidence': 0
        },
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


# main loop to run app in debug mode
if __name__ == '__main__':
    app.run()
    # for production use the line below
    # app.run(debug=False,port=os.getenv('PORT',5000))
