from flask import Flask, jsonify, json, request

import model

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



app.run(port=5000)
