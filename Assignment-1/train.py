import pandas as pd
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

import pickle



def train(train_csv_path,model_name):
    
    train_df = pd.read_csv(train_csv_path,header=None,encoding='utf-8')
    train_df.columns=["Review","Rating"]
    train_df["Rating"]=train_df["Rating"].astype(int)
    train_df.dropna(axis=0,how="any",subset=None,inplace=True)
    train_df["Review"]=train_df["Review"].astype(str)
    
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')

    STOP_WORDS=set(['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldn', 'couldnt', 'cry', 'd', 'de', 'describe', 'detail', 'did', 'didn', 'do', 'does', 'doesn', 'doing', 'don', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'hadn', 'has', 'hasn', 'hasnt', 'have', 'haven', 'having', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'isn', 'it', 'its', 'itself', 'just', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'll', 'ltd', 'm', 'ma', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mightn', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'mustn', 'my', 'myself', 'name', 'namely', 'needn', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'shan', 'she', 'should', 'shouldn', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 't', 'take', 'ten', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 've', 'very', 'via', 'was', 'wasn', 'we', 'well', 'were', 'weren', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'won', 'would', 'wouldn', 'y', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves'])
    NEG_STOP_WORDS=set(['aint', 'cannot', 'cant', 'darent', 'didnt', 'doesnt', 'dont', 'hadnt', 'hardly', 'hasnt', 'havent', 'havnt', 'isnt', 'lack', 'lacking', 'lacks', 'neither', 'never', 'no', 'nobody', 'none', 'nor', 'not', 'nothing', 'nowhere', 'mightnt', 'mustnt', 'neednt', 'oughtnt', 'shant', 'shouldnt', 'wasnt', 'without', 'wouldnt', 'ain', 'can', 'daren', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'havn', 'isn', 'mightn', 'mustn', 'needn', 'oughtn', 'shan', 'shouldn', 'wasn', 'wouldn','not'])
    
    FIN_STOP_WORDS=list(STOP_WORDS-NEG_STOP_WORDS)

    cv = CountVectorizer(stop_words=FIN_STOP_WORDS,ngram_range = (1,2),tokenizer = token.tokenize,max_features=6250)
    # cv = TfidfVectorizer(stop_words=FIN_STOP_WORDS,ngram_range = (1,2),tokenizer = token.tokenize,max_features=6250)

    
    text_counts = cv.fit_transform(train_df['Review'])

    X,Y= text_counts,train_df["Rating"]
   

    model = MultinomialNB()
    # model = LogisticRegression(max_iter=1000,class_weight="balanced",n_jobs=-1)
    model.fit(X,Y)
    print(model.score(X,Y))

    with open(model_name, 'wb') as fout:
        pickle.dump((model,cv), fout)


if __name__ == "__main__":
    train_csv_path=sys.argv[1]
    model_name=sys.argv[2]
    train(train_csv_path,model_name)


    


