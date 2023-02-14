import pandas as pd
import sys
import pickle



def test(model_name,test_csv_path,output_csv):
    
    test_df = pd.read_csv(test_csv_path,header=None,encoding='utf-8')
    test_df.columns=["Review"]    
    with open(model_name, 'rb') as f:
        loaded_model,cv = pickle.load(f)
    text_counts = cv.transform(test_df['Review'])

    X_test= text_counts   
    result = loaded_model.predict(X_test)
    output_df = pd.DataFrame(result)
    output_df.to_csv(output_csv,index=False, header=False)





if __name__ == "__main__":
    model_name=sys.argv[1]
    test_csv_path=sys.argv[2]
    output_csv=sys.argv[3]
    test(model_name,test_csv_path,output_csv)


    


