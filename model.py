
from data import fn_load_final_dataframe

def getAnswers(question):
    print(question)

    df_final = fn_load_final_dataframe()
    
    df_final['q'] = question

    df_final.to_csv('csv/query.csv', index=False)



    answers = [
            {'ans': 'Mental health is important', 'prob': 0.9},
            {'ans': 'Mental health is often neglected', 'prob': 0.8}
            ]

    return answers