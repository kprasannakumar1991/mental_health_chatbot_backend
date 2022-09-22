from flask import Flask, jsonify, json, request

import model
import data
import statistical_model

app = Flask(__name__)

# GET url:port/
@app.route('/')
def getHomePage():
    responseObj = {
        'msg': 'Welcome to Mental Health chat bot',
        'code': 1
    }

    return jsonify(responseObj)

# POST url:port/question
@app.route('/question', methods=['POST'])
def postQuestion():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        body = request.get_json()
        question = body['question']
        answers = model.getAnswers(question)
        
        responseObj = {
            'question': question,
            'answers': answers
        }

        return jsonify(responseObj)
        
    else:
        return 'Content-Type not supported'


# GET /test1
@app.route('/test1')
def testing1():
    
    statistical_model.fn_create_and_save_model1()
    statistical_model.fn_create_and_save_model2()

    return 'done'

# GET /test2
@app.route('/test2')
def testing2():
    
    statistical_model.fn_load_model()

    return 'done'

app.run(port=5000)
