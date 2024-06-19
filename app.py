from gradio_client import Client, handle_file
from flask import Flask, jsonify, make_response, render_template
from flask import request
from flask_cors import CORS
from flask_socketio import SocketIO, send, disconnect
import json
import re
import string

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

image_client = Client("Ritesh-hf/Inappropriate-Image-Classification-using-ViT")
text_client = Client("Ritesh-hf/Inappropriate-Text-Classifier")

def predict_image(img_url_list):
    for img_url in img_url_list:
        print(img_url)
        if not img_url.endswith(".svg"):
            try:
                result = image_client.predict(image=handle_file(img_url), api_name="/classify_image")
                if result["label"] == "Inappropriate":
                    prediction = dict()
                    prediction["type"] = "image"
                    prediction["url"] = img_url
                    prediction["label"] = result["label"]
                    send(prediction)
            except Exception as e:
                print(e)
    return

# def clean_text(text):
#     # Remove tabs and newlines
#     text = text.replace('\t', ' ').replace('\n', ' ')
#     # Remove extra spaces
#     text = re.sub(' +', ' ', text)
#     # Remove URLs
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#     # Remove special characters (retain letters, digits, and basic punctuation)
#     text = re.sub(f'[^{re.escape(string.ascii_letters + string.digits + ".,;!?() ")}]', '', text)
#     # Remove emojis
#     emoji_pattern = re.compile(
#         "["u"\U0001F600-\U0001F64F"  # emoticons
#         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#         u"\U0001F700-\U0001F77F"  # alchemical symbols
#         u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
#         u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
#         u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
#         u"\U0001FA00-\U0001FA6F"  # Chess Symbols
#         u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
#         u"\U00002702-\U000027B0"  # Dingbats
#         u"\U000024C2-\U0001F251"
#         "]+", flags=re.UNICODE
#     )
#     text = emoji_pattern.sub(r'', text)
#     # Remove numbers
#     text = re.sub(r'\d+', '', text)
#     # Remove extra spaces again after all processing
#     text = re.sub(' +', ' ', text)
#     return text.strip()

def predict_text(texts):
    try:
        for text in texts:
            result = text_client.predict(text=text, api_name="/classify_text")
            if result['label'] == "Inappropriate":
                prediction = dict()
                prediction["type"] = "text"
                prediction["text"] = text
                prediction["label"] = result["label"]
                send(prediction)
    except Exception as e:
        print(e)
    return

@app.route('/api/v1/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'POST':
        print("request received")
        img_url_list = request.json["images"]
        print(img_url_list)
        # texts = request.json["texts"]
        # texts_outputs = predict_text(texts)
        image_outputs = predict_image(img_url_list)
        return jsonify(image_outputs)

    if request.method == 'OPTIONS':
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Max-Age"] = "3600"
        return response

@socketio.on('message')
def handle_message(data):
    img_url_list = data["images"]
    texts = data["texts"]
    predict_text(texts)
    predict_image(img_url_list)
    send("Completed")
    # send(f'Echo: {msg}', broadcast=True)


if __name__ == '__main__':
    socketio.run(app, debug=True)
