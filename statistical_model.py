import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from data import fn_split_data
from utils.fuzzy import fn_common_words_count, fn_fuzz_partial_ratio, fn_fuzz_ratio, fn_fuzz_token_set_ratio, fn_fuzz_token_sort_ratio, jaccard_similarity


def fn_create_statistical_features_and_split_data():
    
    # load tr, eval and ts datasets 
    train_df, eval_df, ts_df = fn_split_data()

    # Train
    fuzz_ratio_tr = train_df.apply(fn_fuzz_ratio, axis=1)
    fuzz_partial_ratio_tr = train_df.apply(fn_fuzz_partial_ratio, axis=1)
    fuzz_token_sort_ratio_tr = train_df.apply(fn_fuzz_token_sort_ratio, axis=1)
    fuzz_token_set_ratio_tr = train_df.apply(fn_fuzz_token_set_ratio, axis=1)
    common_words_count_tr = train_df.apply(fn_common_words_count, axis=1)
    js_tr = train_df.apply(jaccard_similarity, axis=1)
    kw = dict(f1 = fuzz_ratio_tr,
            f2 = fuzz_token_sort_ratio_tr,
            f3 = fuzz_partial_ratio_tr,
            f4 = fuzz_token_set_ratio_tr,
            f5 = common_words_count_tr,
            f6 = js_tr,
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
    kw = dict(f1 = fuzz_ratio_eval,
            f2 = fuzz_token_sort_ratio_eval,
            f3 = fuzz_partial_ratio_eval,
            f4 = fuzz_token_set_ratio_eval,
            f5 = common_words_count_eval,
            f6 = js_eval,
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


    kw = dict(f1 = fuzz_ratio_ts,
            f2 = fuzz_token_sort_ratio_ts,
            f3 = fuzz_partial_ratio_ts,
            f4 = fuzz_token_set_ratio_ts,
            f5 = common_words_count_ts,
            f6 = js_ts,
            labels = ts_df.y.values
            )
    statistical_df_ts = pd.DataFrame().assign(**kw) # 4 ratios + 1 + y = 6 columns

    return statistical_df_tr, statistical_df_eval, statistical_df_ts

    

    # print the mean and std-devs of statistical features
    # print(scaler.n_features_in_)
    
def fn_create_and_save_model1():
        statistical_df_tr, statistical_df_eval, statistical_df_ts = fn_create_statistical_features_and_split_data()
        # Standaridze the dataset
        scaler = StandardScaler()
        scaler.fit(statistical_df_tr.iloc[:, :-1])

        statistical_df_tr.iloc[:, :-1] = scaler.transform(statistical_df_tr.iloc[:, :-1])
        statistical_df_eval.iloc[:, :-1] = scaler.transform(statistical_df_eval.iloc[:, :-1])
        statistical_df_ts.iloc[:, :-1] = scaler.transform(statistical_df_ts.iloc[:, :-1])

        df_mean_std = pd.DataFrame().assign(means=scaler.mean_, stds=scaler.var_ ** 0.5)
        df_mean_std.to_csv('csv/statistical_means_stds1.csv', index=False)

        # Saving the StandardScaler model
        joblib.dump(scaler, 'pickle/scaler1.pkl')

        # X_tr = statistical_df_tr.iloc[:, :-1].values
        # y_tr = statistical_df_tr.iloc[:, -1].values

        # model = SVC(C=1000, gamma=0.01, kernel='sigmoid', probability=True)
        # model.fit(X_tr, y_tr)
        # joblib.dump(model, 'model.pkl')

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


def fn_create_and_save_model2():
        statistical_df_tr, statistical_df_eval, statistical_df_ts = fn_create_statistical_features_and_split_data()
        # Standaridze the dataset
        scaler = StandardScaler()
        scaler.fit(statistical_df_tr.values[:, :-1])

        statistical_df_tr.values[:, :-1] = scaler.transform(statistical_df_tr.values[:, :-1])
        statistical_df_eval.values[:, :-1] = scaler.transform(statistical_df_eval.values[:, :-1])
        statistical_df_ts.values[:, :-1] = scaler.transform(statistical_df_ts.values[:, :-1])

        df_mean_std = pd.DataFrame().assign(means=scaler.mean_, stds=scaler.var_ ** 0.5)
        df_mean_std.to_csv('csv/statistical_means_stds2.csv', index=False)

        # Saving the StandardScaler model
        joblib.dump(scaler, 'pickle/scaler2.pkl')

       

def fn_load_model():
        scaler1 = joblib.load('pickle/scaler1.pkl')
        print(scaler1.mean_)
        print(scaler1.var_ ** 0.5)

        print("*************")
        scaler2 = joblib.load('pickle/scaler2.pkl')
        print(scaler2.mean_)
        print(scaler2.var_ ** 0.5)

