from flask import Flask, request, jsonify
import os
import traceback
import json

from rafiki.model import deserialize_knob_config
from rafiki.constants import UserType
from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.utils.auth import generate_token, decode_token, UnauthorizedError, auth

from .service import AdvisorService

service = AdvisorService()

app = Flask(__name__)

@app.route('/')
def index():
    return 'Rafiki Advisor is up.'

@app.route('/advisors', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def create_advisor(auth):
    params = get_request_params()

    # Deserialize knob config
    if 'knob_config_str' in params:
        params['knob_config'] = deserialize_knob_config(params['knob_config_str'])
        del params['knob_config_str']

    return jsonify(service.create_advisor(**params))

@app.route('/advisors/<advisor_id>/propose', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def generate_proposal(auth, advisor_id):
    params = get_request_params()
    return jsonify(service.generate_proposal(advisor_id, **params))

@app.route('/advisors/<advisor_id>/feedback', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def feedback(auth, advisor_id):
    params = get_request_params()
    return jsonify(service.feedback(advisor_id, **params))

@app.route('/advisors/<advisor_id>', methods=['DELETE'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def delete_advisor(auth, advisor_id):
    params = get_request_params()
    return jsonify(service.delete_advisor(advisor_id, **params))

# Handle uncaught exceptions with a server error & the error's stack trace (for development)
@app.errorhandler(Exception)
def handle_error(error):
    return traceback.format_exc(), 500

# Extract request params from Flask request
def get_request_params():
    # Get params from body as JSON
    params = request.get_json()

    # If the above fails, get params from body as form data
    if params is None:
        params = request.form.to_dict()

    # Merge in query params
    query_params = {
        k: v
        for k, v in request.args.items()
    }
    params = {**params, **query_params}

    return params