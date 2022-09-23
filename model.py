
import pandas as pd
import joblib
from utils.text_preprocessing import  remove_special_characters
from utils.fuzzy import fn_common_words_count, fn_fuzz_partial_ratio, fn_fuzz_ratio, fn_fuzz_token_set_ratio, fn_fuzz_token_sort_ratio, jaccard_similarity

def getAnswers(question):
    print(question)

    question = remove_special_characters(question)

    df_final = pd.read_csv('csv/mental_health_faq_final.csv')
    df_final['q'] = question
    df_final.to_csv('csv/query.csv', index=False)

    fn_create_features_for_query()

    answers = [
            {'ans': 'Mental health is important', 'prob': 0.9},
            {'ans': 'Mental health is often neglected', 'prob': 0.8}
            ]

    return answers


def fn_create_features_for_query():
        df_query = pd.read_csv('csv/query.csv')

        fuzz_ratio_tr = df_query.apply(fn_fuzz_ratio, axis=1)
        fuzz_partial_ratio_tr = df_query.apply(fn_fuzz_partial_ratio, axis=1)
        fuzz_token_sort_ratio_tr = df_query.apply(fn_fuzz_token_sort_ratio, axis=1)
        fuzz_token_set_ratio_tr = df_query.apply(fn_fuzz_token_set_ratio, axis=1)
        common_words_count_tr = df_query.apply(fn_common_words_count, axis=1)
        js_tr = df_query.apply(jaccard_similarity, axis=1)
        kw = dict(f1 = fuzz_ratio_tr,
            f2 = fuzz_token_sort_ratio_tr,
            f3 = fuzz_partial_ratio_tr,
            f4 = fuzz_token_set_ratio_tr,
            f5 = common_words_count_tr,
            f6 = js_tr,
            labels = df_query.y.values
            )

        statistical_df = pd.DataFrame().assign(**kw)

        statistical_df.to_csv('csv/query_statistical.csv')

        # load the StandardScaler model
        scaler = joblib.load('pickle/scaler1.pkl')
        print('StandScaler')
        print(scaler.mean_)
        print(scaler.var_)

        # Transform the values to standard ones
        statistical_df.values[:, :-1] = scaler.transform(statistical_df.values[:, :-1])
        statistical_df.to_csv('csv/query_statistical_std.csv')




        