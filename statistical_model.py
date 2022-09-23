import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from utils.fuzzy import fn_common_words_count, fn_fuzz_partial_ratio, fn_fuzz_ratio, fn_fuzz_token_set_ratio, fn_fuzz_token_sort_ratio, jaccard_similarity
from utils import mlfuncs_clfn as ml
from utils.text_preprocessing import  remove_special_characters

def fn_create_statistical_features():
    random.seed(10)
    df_final = pd.read_csv('csv/mental_health_faq_final.csv')

    # features for tr, eval and ts sets
    train_df, eval_df, ts_df = ml.fn_tr_eval_ts_split_clf(df_final, eval_size = 0.1, ts_size = 0.1)

    # Train
    fuzz_ratio_tr = train_df.apply(fn_fuzz_ratio, axis=1)
    fuzz_partial_ratio_tr = train_df.apply(fn_fuzz_partial_ratio, axis=1)
    fuzz_token_sort_ratio_tr = train_df.apply(fn_fuzz_token_sort_ratio, axis=1)
    fuzz_token_set_ratio_tr = train_df.apply(fn_fuzz_token_set_ratio, axis=1)
    common_words_count_tr = train_df.apply(fn_common_words_count, axis=1)
    js_tr = train_df.apply(jaccard_similarity, axis=1)
    kw = dict(
        #     f1 = fuzz_ratio_tr,
        #     f2 = fuzz_token_sort_ratio_tr,
            f3 = fuzz_partial_ratio_tr,
            f4 = fuzz_token_set_ratio_tr,
            f5 = common_words_count_tr,
        #     f6 = js_tr,
            labels = train_df.y.values
            )
    statistical_df_tr = pd.DataFrame().assign(**kw) # 4 ratios + 1 + y = 6 columns


    # Eval
    fuzz_ratio_eval = eval_df.apply(fn_fuzz_ratio, axis=1)
    fuzz_partial_ratio_eval = eval_df.apply(fn_fuzz_partial_ratio, axis=1)
    fuzz_token_sort_ratio_eval = eval_df.apply(fn_fuzz_token_sort_ratio, axis=1)
    fuzz_token_set_ratio_eval = eval_df.apply(fn_fuzz_token_set_ratio, axis=1)
    common_words_count_eval = eval_df.apply(fn_common_words_count, axis=1)
    js_eval = eval_df.apply(jaccard_similarity, axis=1)
    kw = dict(
        #     f1 = fuzz_ratio_eval,
        #     f2 = fuzz_token_sort_ratio_eval,
            f3 = fuzz_partial_ratio_eval,
            f4 = fuzz_token_set_ratio_eval,
            f5 = common_words_count_eval,
        #     f6 = js_eval,
            labels = eval_df.y.values
            )
    statistical_df_eval = pd.DataFrame().assign(**kw) # 4 ratios + 1 + y = 6 columns


    # Test
    fuzz_ratio_ts = ts_df.apply(fn_fuzz_ratio, axis=1)
    fuzz_partial_ratio_ts = ts_df.apply(fn_fuzz_partial_ratio, axis=1)
    fuzz_token_sort_ratio_ts = ts_df.apply(fn_fuzz_token_sort_ratio, axis=1)
    fuzz_token_set_ratio_ts = ts_df.apply(fn_fuzz_token_set_ratio, axis=1)
    common_words_count_ts = ts_df.apply(fn_common_words_count, axis=1)
    js_ts = ts_df.apply(jaccard_similarity, axis=1)


    kw = dict(
        #     f1 = fuzz_ratio_ts,
        #     f2 = fuzz_token_sort_ratio_ts,
            f3 = fuzz_partial_ratio_ts,
            f4 = fuzz_token_set_ratio_ts,
            f5 = common_words_count_ts,
        #     f6 = js_ts,
            labels = ts_df.y.values
            )
    statistical_df_ts = pd.DataFrame().assign(**kw) # 4 ratios + 1 + y = 6 columns

    return statistical_df_tr, statistical_df_eval, statistical_df_ts

    

    # print the mean and std-devs of statistical features
    # print(scaler.n_features_in_)
    
def fn_create_pure_model():
        random.seed(10)
        df_final = pd.read_csv('csv/mental_health_faq_final.csv')
        df_copy = df_final.copy()
        # features (used to testing purpose)
        fuzz_ratio_tr = df_copy.apply(fn_fuzz_ratio, axis=1)
        fuzz_partial_ratio_tr = df_copy.apply(fn_fuzz_partial_ratio, axis=1)
        fuzz_token_sort_ratio_tr = df_copy.apply(fn_fuzz_token_sort_ratio, axis=1)
        fuzz_token_set_ratio_tr = df_copy.apply(fn_fuzz_token_set_ratio, axis=1)
        common_words_count_tr = df_copy.apply(fn_common_words_count, axis=1)
        js_tr = df_copy.apply(jaccard_similarity, axis=1)
        kw = dict(
                # f1 = fuzz_ratio_tr,
                # f2 = fuzz_token_sort_ratio_tr,
                f3 = fuzz_partial_ratio_tr,
                f4 = fuzz_token_set_ratio_tr,
                f5 = common_words_count_tr,
                # f6 = js_tr,
                labels = df_copy.y.values
                )
        df_features = pd.DataFrame().assign(**kw) # 4 ratios + 1 + y = 6 columns
        # df_features.to_csv('csv/features.csv', index=False)
        scaler = StandardScaler()
        scaler.fit(df_features.iloc[:, :-1])
        df_features.iloc[:, :-1] = scaler.transform(df_features.iloc[:, :-1])
        # df_features.to_csv('csv/features_std.csv', index=False)
        joblib.dump(scaler, 'pickle/scaler.pkl')

        X_tr = df_features.iloc[:, :-1].values
        y_tr = df_features.iloc[:, -1].values

        model = SVC(C=1000, gamma=0.01, kernel='sigmoid', probability=True)
        model.fit(X_tr, y_tr)
        joblib.dump(model, 'pickle/model.pkl')

        
def fn_create_statistical_model():
        statistical_df_tr, statistical_df_eval, statistical_df_ts = fn_create_statistical_features()
        # Standaridze the dataset
        scaler = StandardScaler()
        scaler.fit(statistical_df_tr.iloc[:, :-1])

        statistical_df_tr.iloc[:, :-1] = scaler.transform(statistical_df_tr.iloc[:, :-1])
        statistical_df_eval.iloc[:, :-1] = scaler.transform(statistical_df_eval.iloc[:, :-1])
        statistical_df_ts.iloc[:, :-1] = scaler.transform(statistical_df_ts.iloc[:, :-1])
        # statistical_df_tr.to_csv('csv/statistical_tr_std.csv', index=False)

        # df_mean_std = pd.DataFrame().assign(means=scaler.mean_, stds=scaler.var_ ** 0.5)
        # df_mean_std.to_csv('csv/statistical_means_stds1.csv', index=False)

        # Saving the StandardScaler model
        joblib.dump(scaler, 'pickle/scaler.pkl')

        X_tr = statistical_df_tr.iloc[:, :-1].values
        y_tr = statistical_df_tr.iloc[:, -1].values

        model = SVC(C=1000, gamma=0.01, kernel='sigmoid', probability=True)
        model.fit(X_tr, y_tr)
        joblib.dump(model, 'pickle/model.pkl')

        # features = [column for column in statistical_df_tr.columns.values if column is not 'labels']
        # means = []
        # stds = []
        # for column in features:
        #     scaler = StandardScaler()
        #     scaler.fit(np.array(statistical_df_tr[column]).reshape(-1, 1))
        #     statistical_df_tr[column] = scaler.transform(np.array(statistical_df_tr[column]).reshape(-1, 1))
        #     statistical_df_eval[column] = scaler.transform(np.array(statistical_df_eval[column]).reshape(-1, 1))
        #     statistical_df_ts[column] = scaler.transform(np.array(statistical_df_ts[column]).reshape(-1, 1))
        #     means.append(scaler.mean_[0])
        #     stds.append(scaler.var_[0] ** 0.5)

        # df_mean_std = pd.DataFrame().assign(means=means, stds=stds)
        # df_mean_std.to_csv('csv/statistical_means_stds.csv', index=False)

        # statistical_df_tr.to_csv('csv/statistical_df_tr.csv', index = False)
        # statistical_df_eval.to_csv('csv/statistical_df_eval.csv', index = False)
        # statistical_df_ts.to_csv('csv/statistical_df_ts.csv', index = False)

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

        question = remove_special_characters(question)
        df = pd.read_csv('csv/mental_health_faq_final.csv')

        df = df[df['y']==1]
        df['q'] = question

        # df.to_csv('csv/query.csv', index=False)

        fuzz_ratio_tr = df.apply(fn_fuzz_ratio, axis=1)
        fuzz_partial_ratio_tr = df.apply(fn_fuzz_partial_ratio, axis=1)
        fuzz_token_sort_ratio_tr = df.apply(fn_fuzz_token_sort_ratio, axis=1)
        fuzz_token_set_ratio_tr = df.apply(fn_fuzz_token_set_ratio, axis=1)
        common_words_count_tr = df.apply(fn_common_words_count, axis=1)
        js_tr = df.apply(jaccard_similarity, axis=1)
        kw = dict(
        #     f1 = fuzz_ratio_tr,
        #     f2 = fuzz_token_sort_ratio_tr,
            f3 = fuzz_partial_ratio_tr,
            f4 = fuzz_token_set_ratio_tr,
            f5 = common_words_count_tr,
        #     f6 = js_tr
            )
        df_features = pd.DataFrame().assign(**kw) # 4 ratios + 1 + y = 6 columns
        # df_features.to_csv('csv/query_features.csv', index=False)

        df_features.iloc[:,:] = scaler.transform(df_features.iloc[:, :])
        # df_features.to_csv('csv/query_features_std.csv', index=False)

        print("______________________________")

        # invalid pair probability
        invalid_pair_probs = model.predict_proba(df_features)[:, 0]
        invalid_prob = np.max(invalid_pair_probs)
        print('Max Invalid prob ', invalid_prob)
        top_3 = np.sort(invalid_pair_probs)[::-1][:3]
        print(top_3)

        print("______________________________")
        # valid pair probability
        valid_pair_probs = model.predict_proba(df_features)[:, 1]
        valid_prob = np.max(valid_pair_probs)
        print('Max Valid prob ', valid_prob)
        top_3 = np.sort(valid_pair_probs)[::-1][:3]
        print(top_3)
        indices = np.argsort(valid_pair_probs)[::-1][:3]
        print(indices)
        valid_question = 0
        top_answers = []

        print("______________________________")

        # 2.5% margin is added
        if valid_prob + 0.20 > invalid_prob:
                valid_question = 1
                for i in indices:
                        ans = {
                                'ans': df.iloc[i].a[: 50], 
                                'probability': valid_pair_probs[i],
                                'q_no': str(i),
                                'line_no_in_csv': str(i+2)}
                        top_answers.append(ans)


        return valid_question, top_answers
