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
        valid_question, top_answers = statistical_model.fn_find_answer(question)
        
        resposneObj = {'valid_question': valid_question, 'answers': top_answers}
        return jsonify(resposneObj)
        
    else:
        return 'Content-Type not supported'



# GET /createDataset
@app.route('/createDataset')
def createDataset():
    data.fn_create_dataset_for_ml()
    
    return 'dataset final_csv created'

# GET /test1
@app.route('/createStatisticalModel')
def createStatisticalModel():
    
    # statistical_model.fn_create_pure_model()
    statistical_model.fn_create_statistical_model()

    return 'Statistical Model and StandardScaler Model created'

# GET /test2
@app.route('/loadStatisticalModel')
def loadStatisticalModel():
    statistical_model.fn_load_statistical_model()

    return 'done'

app.run(port=5000, debug=True)
