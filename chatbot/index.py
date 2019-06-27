from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from db import db, TrainAppName, InferenceAppName
import os
import dialogflow
import requests
import json
import sys
sys.path.insert(0, '..')
from rafiki.client import Client
# import pusher

db_filename = "chatbot.db"
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///%s' % db_filename
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = True

client = Client(admin_host='localhost', admin_port=3000)

db.init_app(app)
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

# run Flask app
if __name__ == "__main__":
    load_dotenv()
    app.run()

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json(silent=True)
    if data['queryResult']['intent']['displayName'] == "train_job - naming":
        name = data['queryResult']['queryText']
        app_name = TrainAppName(name=name)
        db.session.add(app_name)
        db.session.commit()
        reply = {
            "fulfillmentText": "Your app is called {0}. Where is your training dataset located?".format(name),
        }
        return jsonify(reply)
    elif data['queryResult']['intent']['displayName'] == "train_job - testing dataset":
        names = TrainAppName.query.all()
        name = names[0].serialize()['name']
        client.login(email='superadmin@rafiki', password='rafiki')
        client.create_train_job(
            app=name,
            task=data['queryResult']['outputContexts'][-1]['parameters']['task'],
            train_dataset_uri=data['queryResult']['outputContexts'][1]['parameters']['train_url'],
            test_dataset_uri=data['queryResult']['outputContexts'][0]['parameters']['test_url'],
            budget={ 'MODEL_TRIAL_COUNT': 5 }
            )
        reply = {
            "fulfillmentText": ('The training starts. Please wait for several minutes and then you can log ' 
                                'into http://127.0.0.1:3001 to see the training progress.'),
        }
        return jsonify(reply)
    elif data['queryResult']['intent']['displayName'] == "inference_job - name":
        name = data['queryResult']['queryText']
        app_name = InferenceAppName(name=name)
        db.session.add(app_name)
        db.session.commit()
        app_version = data['queryResult']['outputContexts'][-1]['parameters']['app_version']
        app_version = int(app_version)
        client.login(email='superadmin@rafiki', password='rafiki')
        client.create_inference_job(app=name, app_version=app_version)
        reply = {
            "fulfillmentText": ('Your inference job is being set up. Please wait for about '
                                'a minute and then you can request the inference server address.'),
        }
        return jsonify(reply)
    elif data['queryResult']['intent']['displayName'] == "inference_job - address":
        names = InferenceAppName.query.all()
        name = names[0].serialize()['name']
        app_version = data['queryResult']['outputContexts'][-1]['parameters']['app_version']
        app_version = int(app_version)
        inference_data = client.get_running_inference_job(app=name, app_version=app_version)
        address = inference_data['predictor_host']
        reply = {
            "fulfillmentText": "the address of the inference job is {0}".format(address),
        }
        return jsonify(reply)

def detect_intent_texts(project_id, session_id, text, language_code):
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(project_id, session_id)

        if text:
            text_input = dialogflow.types.TextInput(
                text=text, language_code=language_code)
            query_input = dialogflow.types.QueryInput(text=text_input)
            response = session_client.detect_intent(
                session=session, query_input=query_input)

            return response.query_result.fulfillment_text

def send_message():
    message = request.form['message']
    project_id = os.getenv('DIALOGFLOW_PROJECT_ID')
    fulfillment_text = detect_intent_texts(project_id, "unique", message, 'en')
    response_text = { "message":  fulfillment_text }
    return jsonify(response_text)
