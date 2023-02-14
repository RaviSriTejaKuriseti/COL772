import pandas as pd


test_df = pd.read_csv("./datasets/val_split.csv",header=None,encoding='utf-8')
test_df.drop(test_df.columns[[0]], axis=1, inplace=True)
test_df.to_csv("val_output.csv",index=False, header=False)



    


