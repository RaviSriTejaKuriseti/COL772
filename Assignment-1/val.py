import pandas as pd
import sys
import pickle
from sklearn.metrics import f1_score



def validate(model_name,val_csv_path):
    
    test_df = pd.read_csv(val_csv_path,header=None,encoding='utf-8')
    test_df.columns=["Review","Rating"]    
    with open(model_name, 'rb') as f:
        loaded_model,cv = pickle.load(f)
    text_counts = cv.transform(test_df['Review'])

    X_val= text_counts
    Y_val=test_df["Rating"]

    Y_observed=loaded_model.predict(X_val)

    f1_micro = f1_score(Y_val, Y_observed, average='micro')
    f1_macro = f1_score(Y_val, Y_observed, average='macro')

    print(f1_micro,f1_macro)





if __name__ == "__main__":
    model_name=sys.argv[1]
    val_csv_path=sys.argv[2]
    validate(model_name,val_csv_path)


    


