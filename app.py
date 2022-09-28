from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import model
import utils.data_creation as data_creation
import statistical_model

app = Flask(__name__)
CORS(app)

# Returns HomePage
@app.route('/')
def getHomePage():
    return "<h1>Welcome to Mental health Chat Bot</h1>"

# Creates data set for the model
@app.route('/createDataset')
def createDataset():
    data_creation.fn_create_dataset_for_ml()
    
    return 'dataset final_csv created'


##################### Statistical features ###########################
######################################################################
######################################################################

# Returns the answer(s) for the asked question
# It uses the model created from Statistical features only
@app.route('/question', methods=['POST'])
def postQuestion():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        body = request.get_json()
        question = body['question']
        question, top_answers = statistical_model.fn_find_answer(question)
        
        resposneObj = {'_question': question,'answers': top_answers}
        return jsonify(resposneObj)
        
    else:
        return 'Content-Type not supported'

# Initialize and create a Model
@app.route('/createStatisticalModel')
def createStatisticalModel():    
    statistical_model.fn_create_model()

    return 'Model based on only Statistical features created'

######################################################################
######################################################################


##################### Combined features ##############################
######################################################################
######################################################################
# Initialize and create a Model
@app.route('/createTFIDFModel')
def createTFIDFModel():
    model.fn_create_model()

    return 'Model based on TFIDF & Statistical features created'

# Returns the answer(s) for the asked question
# It uses the model created from TF-IDF & Statistical features combined
@app.route('/question1', methods=['POST'])
def postQuestion1():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        body = request.get_json()
        question = body['question']
        question, top_answers = model.fn_find_answer(question)
        
        resposneObj = {'_question': question,'answers': top_answers}
        return jsonify(resposneObj)
        
    else:
        return 'Content-Type not supported'

######################################################################
######################################################################

if __name__ == '__main__':
    app.run()