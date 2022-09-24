import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from utils.fuzzy import fn_common_words_count, fn_fuzz_partial_ratio, fn_fuzz_ratio, fn_fuzz_token_set_ratio, fn_fuzz_token_sort_ratio, jaccard_similarity
from utils.text_preprocessing import  text_process_question


def fn_create_model():
        random.seed(10)
        df_final = pd.read_csv('csv/mental_health_faq_final.csv')
        df = df_final.copy()
        # features (used to testing purpose)
        fuzz_partial_ratio = df.apply(fn_fuzz_partial_ratio, axis=1)
        fuzz_token_set_ratio = df.apply(fn_fuzz_token_set_ratio, axis=1)
        common_words_count = df.apply(fn_common_words_count, axis=1)
        kw = dict(
                f1 = fuzz_partial_ratio,
                f2 = fuzz_token_set_ratio,
                f3 = common_words_count,
                labels = df.y.values
                )
        df_features = pd.DataFrame().assign(**kw)
        scaler = StandardScaler()
        scaler.fit(df_features.iloc[:, :-1])
        df_features.iloc[:, :-1] = scaler.transform(df_features.iloc[:, :-1])
        joblib.dump(scaler, 'pickle/scaler.pkl')

        X_tr = df_features.iloc[:, :-1].values
        y_tr = df_features.iloc[:, -1].values

        model = SVC(C=1000, gamma=0.01, kernel='sigmoid', probability=True)
        model.fit(X_tr, y_tr)
        joblib.dump(model, 'pickle/model.pkl')


def fn_load_statistical_model():
        scaler = joblib.load('pickle/scaler.pkl')
        print(scaler.mean_)
        print(scaler.var_ ** 0.5)

        df = pd.read_csv('csv/statistical_tr.csv')
        df.iloc[:, :-1] = scaler.transform(df.iloc[:, :-1])
        df.to_csv('csv/statistical_tr_std2.csv', index=False)



def fn_find_answer(question):
        scaler = joblib.load('pickle/scaler.pkl')
        model = joblib.load('pickle/model.pkl')

        question = text_process_question(question)
        df = pd.read_csv('csv/mental_health_faq_final.csv')

        df = df[df['y']==1]
        df['q'] = question

        fuzz_partial_ratio = df.apply(fn_fuzz_partial_ratio, axis=1)
        fuzz_token_set_ratio = df.apply(fn_fuzz_token_set_ratio, axis=1)
        common_words_count = df.apply(fn_common_words_count, axis=1)
        kw = dict(
            f1 = fuzz_partial_ratio,
            f2 = fuzz_token_set_ratio,
            f3 = common_words_count,
            )
        df_features = pd.DataFrame().assign(**kw)

        df_features.iloc[:,:] = scaler.transform(df_features.iloc[:, :])

        print("_____________stats_________________")

        # invalid pair probability
        invalid_pair_probs = model.predict_proba(df_features)[:, 0]
        invalid_prob = np.max(invalid_pair_probs)
        print('Max Invalid prob ', invalid_prob)
        top_probs = np.sort(invalid_pair_probs)[::-1][:5]
        print(top_probs)

        print("_____________stats_________________")
        # valid pair probability
        valid_pair_probs = model.predict_proba(df_features)[:, 1]
        valid_prob = np.max(valid_pair_probs)
        print('Max Valid prob ', valid_prob)
        top_probs = np.sort(valid_pair_probs)[::-1][:5]
        print(top_probs)
        indices = np.argsort(valid_pair_probs)[::-1][:5]
        print(indices)

        print("_____________stats_________________")
        top_answers = []

        if (valid_prob + 0.20 > invalid_prob) or np.max(common_words_count) >= 2:
                for i in indices:
                        ans = {
                                '_q_no': str(i),
                                'question_answer': df.iloc[i].a[: 50], 
                                'correctness_probability': valid_pair_probs[i]
                        }
                                
                        top_answers.append(ans)


        return question, top_answers
