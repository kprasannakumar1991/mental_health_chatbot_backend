import numpy as np
import pandas as pd

from utils.text_preprocessing import remove_special_characters
from utils import mlfuncs_clfn as ml

def fn_create_QA_dataframe():
    dff = pd.read_csv('csv/mental_health_faq.csv')
    dff.columns = [i.lower() for i in dff.columns]
    dff.drop('question_id', inplace=True, axis=1)
    dff.columns = ['q', 'a']
    
    # data cleaning
    dff.q = dff.q.apply(remove_special_characters)
    dff.a = dff.a.apply(remove_special_characters)

    # creating label for valid question-answer pairs
    dff = dff.assign(y = lambda row: 1)

    # creating dataframe for invalid question-answer pairs
    df_wrong = fn_create_invalid_pairs(dff)

    # combining valid and invalid pairs to create final dataset
    df_final = pd.concat([dff, df_wrong])

    # Testing
    question_index = 0 # can't be greater than 97
    print(df_final.iloc[question_index]) # valid pair
    print(df_final.iloc[question_index+98]) # invalid pair

    df_final.to_csv('csv/mental_health_faq_final.csv', index=False)
    pass


def fn_create_invalid_pairs(dff):
    import random
    from random import choice

    random.seed(10)

    questions = []
    answers = []
    number_of_valid_rows = len(dff)
    for row in range(0, number_of_valid_rows):
        # Creating a new invalid row
        current_row = dff.iloc[row]
        question = current_row['q']
        # making sure we don't pick the same correct answer in our invalid pair
        answer_index = choice([i for i in range(0, number_of_valid_rows) if i not in [row]])
        answer = dff.iloc[answer_index]['a']
        questions.append(question)
        answers.append(answer)

    # marking the pair as invalid pair (y=0)
    df_wrong = pd.DataFrame().assign(q=questions, a=answers, y = 0)

    return df_wrong


def fn_load_final_dataframe():
    df_final = pd.read_csv('csv/mental_health_faq_final.csv')

    return df_final
        

def fn_split_data():
    df_final = pd.read_csv('csv/mental_health_faq_final.csv')

    df_tr, df_eval, df_ts = ml.fn_tr_eval_ts_split_clf(df_final, eval_size = 0.2, ts_size = 0.2)

    return df_tr, df_eval, df_ts



