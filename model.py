
import numpy as np
import pandas as pd
import random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils.text_preprocessing import text_process_question
from utils.statistical_features import fn_common_words_count, fn_fuzz_partial_ratio, fn_fuzz_ratio, fn_fuzz_token_set_ratio, fn_fuzz_token_sort_ratio, jaccard_similarity


def fn_create_model():
    random.seed(10)

    df_final = pd.read_csv('csv/mental_health_faq_final.csv')
    df = df_final.copy()

    # Create statistical features
    fuzz_partial_ratio = df.apply(fn_fuzz_partial_ratio, axis=1)
    fuzz_token_set_ratio = df.apply(fn_fuzz_token_set_ratio, axis=1)
    common_words_count = df.apply(fn_common_words_count, axis=1)

    # Create TDIDF features
    # questions
    tfidf_questions_vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1,3))
    tfidf_questions_vectorizer.fit(df.q.values)
    tfidf_q = tfidf_questions_vectorizer.transform(df.q.values)
    # answers
    tfidf_answers_vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1,3))
    tfidf_answers_vectorizer.fit(df.a.values)
    tfidf_a = tfidf_answers_vectorizer.transform(df.a.values)
    # dot-product between questions and answers
    tfidf_qa_dp = [i @ j for i, j in zip(tfidf_q.A, tfidf_a.A)]
    
    # # combining statistical and tfidf features together
    concat_qa =  np.concatenate([tfidf_q.A, tfidf_a.A], axis = 1)
    kw = dict(
        f1 = fuzz_partial_ratio, # 1
        f2 = fuzz_token_set_ratio, # 1
        f3 = common_words_count, # 1
        f4 = tfidf_qa_dp, #1
        labels = df.y.values
        )

    df_features = pd.DataFrame().assign(**kw)

    # need to standardize the features
    scaler = StandardScaler()
    scaler.fit(df_features.iloc[:, :-1])

    # Training the model
    # transform all features before training the model
    df_features.iloc[:, :-1] = scaler.transform(df_features.iloc[:, :-1])
    X_tr = df_features.iloc[:, :-1].values
    y_tr = df_features.iloc[:, -1].values
    model = SVC(C=1000, gamma=0.01, kernel='sigmoid', probability=True)
    model.fit(X_tr, y_tr)

    # Saving vectorizers, standardrizer, models
    joblib.dump(tfidf_questions_vectorizer, 'pickle/tfidf_questions_vectorizer.pkl')
    joblib.dump(tfidf_answers_vectorizer, 'pickle/tfidf_answers_vectorizer.pkl')
    joblib.dump(scaler, 'pickle/standard_scaler.pkl')
    joblib.dump(model, 'pickle/model1.pkl')


def fn_find_answer(question):
    # load vectorizers, standardizer, model
    tfidf_questions_vectorizer = joblib.load('pickle/tfidf_questions_vectorizer.pkl')
    tfidf_answers_vectorizer = joblib.load('pickle/tfidf_answers_vectorizer.pkl')
    scaler = joblib.load('pickle/standard_scaler.pkl')
    model = joblib.load('pickle/model1.pkl')

    # clean the input question
    question = text_process_question(question)

    # load the dataset
    df = pd.read_csv('csv/mental_health_faq_final.csv')
    df = df[df['y']==1]
    df['q'] = question
   
    # for the asked question create the required features
    # statistical features
    fuzz_partial_ratio = df.apply(fn_fuzz_partial_ratio, axis=1)
    fuzz_token_set_ratio = df.apply(fn_fuzz_token_set_ratio, axis=1)
    common_words_count = df.apply(fn_common_words_count, axis=1)
    # tfidf features
    tfidf_q = tfidf_questions_vectorizer.transform(df.q.values)
    tfidf_a = tfidf_answers_vectorizer.transform(df.a.values)
    tfidf_qa_dp = [i @ j for i, j in zip(tfidf_q.A, tfidf_a.A)]
     # combining statistical and tfidf features together
    concat_qa =  np.concatenate([tfidf_q.A, tfidf_a.A], axis = 1)
    kw = dict(
        f1 = fuzz_partial_ratio, # 1
        f2 = fuzz_token_set_ratio, # 1
        f3 = common_words_count, # 1
        f4 = tfidf_qa_dp #1
        )
    df_features = pd.DataFrame().assign(**kw)

    #standardize all the features
    df_features.iloc[:,:] = scaler.transform(df_features.iloc[:, :])

    # Predicting using the stored model
    print("_____________tfidf_________________")
    # invalid pair probability
    invalid_pair_probs = model.predict_proba(df_features)[:, 0]
    invalid_prob = np.max(invalid_pair_probs)
    print('Max Invalid prob ', invalid_prob)
    top_probs = np.sort(invalid_pair_probs)[::-1][:5]
    print(top_probs)

    print("_____________tfidf_________________")
    # valid pair probability
    valid_pair_probs = model.predict_proba(df_features)[:, 1]
    valid_prob = np.max(valid_pair_probs)
    print('Max Valid prob ', valid_prob)
    top_probs = np.sort(valid_pair_probs)[::-1][:5]
    print(top_probs)
    indices = np.argsort(valid_pair_probs)[::-1][:5]
    print(indices)

    print("_____________tfidf_________________")
    print('Top common words ', np.max(common_words_count))
    top_answers = []

    if (valid_prob + 0.20 > invalid_prob):
    # if (valid_prob + 0.20 > invalid_prob) or np.max(common_words_count) >= 2:
            for i in indices:
                    ans = {
                                '_q_no': str(i),
                                'question_answer': df.iloc[i].a[: 50], 
                                'correctness_probability': valid_pair_probs[i]
                        }
                    top_answers.append(ans)


    return question, top_answers
