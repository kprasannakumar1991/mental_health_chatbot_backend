
import numpy as np
import pandas as pd
import random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from utils.text_preprocessing import text_process_question
from utils.statistical_features import fn_common_words_count, fn_fuzz_partial_ratio, fn_fuzz_ratio, fn_fuzz_token_set_ratio, fn_fuzz_token_sort_ratio, jaccard_similarity


##############################################################
# NOT USED IN THE APP, JUST PRESENT FOR REFERENCE

# TFIDF and Statistical features are combined, then applied PCA and extracted best featuers.
# The Model is trained over the best features.
##############################################################

def fn_create_model():
    random.seed(10)

    df_final = pd.read_csv('../csv/mental_health_faq_final.csv')
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

    # Total features = 100 + 100 + f1 + f2 + f3 + f4 = 304 features
    df_features = pd.DataFrame(concat_qa.A).assign(**kw)
    y_tr = df_features.iloc[:, -1].values

    # 1. need to standardize the features
    scaler = StandardScaler()
    scaler.fit(df_features.iloc[:, :-1])
    df_features.iloc[:, :-1] = scaler.transform(df_features.iloc[:, :-1])

    # 2. Reduce the features to 100 using PCA
    X_features = df_features.iloc[:, :-1].values
    pca = PCA(n_components=100)
    pca.fit(X_features)
    X_pca_features = pca.transform(X_features)

    # 3. Select the top 10 features for training the ML model
    kbest = SelectKBest(score_func=f_classif, k=50)
    kbest.fit(X_pca_features, y_tr)
    X_tr = kbest.transform(X_pca_features)

    model = SVC(C=1000, gamma=0.01, kernel='sigmoid', probability=True)
    model.fit(X_tr, y_tr)

    # Saving vectorizers, standardrizer, models
    joblib.dump(tfidf_questions_vectorizer, 'tfidf_questions_vectorizer.pkl')
    joblib.dump(tfidf_answers_vectorizer, 'tfidf_answers_vectorizer.pkl')
    joblib.dump(scaler, 'scaler2.pkl')
    joblib.dump(pca, 'pca.pkl')
    joblib.dump(kbest, 'kbest.pkl')
    joblib.dump(model, 'model2.pkl')


def fn_find_answer(question):
    # load vectorizers, standardizer, model
    tfidf_questions_vectorizer = joblib.load('tfidf_questions_vectorizer.pkl')
    tfidf_answers_vectorizer = joblib.load('tfidf_answers_vectorizer.pkl')
    scaler = joblib.load('scaler2.pkl')
    pca = joblib.load('pca.pkl')
    kbest = joblib.load('kbest.pkl');
    model = joblib.load('model2.pkl')

    # clean the input question
    question = text_process_question(question)
     # Used to send back response
    df_original = pd.read_csv('../csv/mental_health_faq.csv');
    # Used by ML 
    df = pd.read_csv('../csv/mental_health_faq_final.csv')

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
    df_features = pd.DataFrame(concat_qa.A).assign(**kw)

    # 1. standardize all the features
    df_features.iloc[:,:] = scaler.transform(df_features.iloc[:, :])

    # 2. Reduce the features to 100 using PCA
    X_features = df_features.iloc[:, :].values
    X_pca_features = pca.transform(X_features)

    # 3. Select the top 10 features for training the ML model
    X_tr = kbest.transform(X_pca_features)


    # Predicting using the stored model
    print("_____________tfidf pca_________________")
    # invalid pair probability
    invalid_pair_probs = model.predict_proba(X_tr)[:, 0]
    invalid_prob = np.max(invalid_pair_probs)
    print('Max Invalid prob ', invalid_prob)
    top_probs = np.sort(invalid_pair_probs)[::-1][:3]
    print(top_probs)

    print("_____________tfidf pca_________________")
    # valid pair probability
    valid_pair_probs = model.predict_proba(X_tr)[:, 1]
    valid_prob = np.max(valid_pair_probs)
    print('Max Valid prob ', valid_prob)
    top_probs = np.sort(valid_pair_probs)[::-1][:3]
    print(top_probs)
    indices = np.argsort(valid_pair_probs)[::-1][:3]
    print(indices)

    print("_____________tfidf pca_________________")
    print('Top common words ', np.max(common_words_count))
    top_answers = []

    if (valid_prob + 0.20 > invalid_prob):
    # if (valid_prob + 0.20 > invalid_prob) or np.max(common_words_count) >= 2:
            for i in indices:
                    ans = {
                            '_q_no': str(i),
                            'question': df_original.iloc[i].Questions,
                            'answer': df_original.iloc[i].Answers, 
                            'probability': round(valid_pair_probs[i] * 100, 2)
                        }
                    top_answers.append(ans)


    return question, top_answers
