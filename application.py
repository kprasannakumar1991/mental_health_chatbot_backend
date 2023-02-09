from flask import Flask, jsonify, request
from flask_cors import CORS
import model
import utils.data_creation as data_creation

application = Flask(__name__)
CORS(application)

# Returns HomePage
@application.route('/')
def getHomePage():
    return "<h1>Welcome to Mental Health Chat Bot</h1>"

# Creates data set for the model
@application.route('/createDataset')
def createDataset():
    data_creation.fn_create_dataset_for_ml()
    
    return 'dataset final_csv created'


# Initialize and create a Model
@application.route('/createModel')
def createModel():
    model.fn_create_model()

    return 'Model based on TFIDF & Statistical features created'

# Returns the answer(s) for the asked question
@application.route('/question1', methods=['POST'])
def question():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        body = request.get_json()
        question = body['question']
        question, top_answers = model.fn_find_answer(question)
        
        resposneObj = {'_question': question,'answers': top_answers}
        return jsonify(resposneObj)
        
    else:
        return 'Content-Type not supported'

# if __name__ == '__main__':
#     application.run()