#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys
import random
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
import argparse
import math
from torch.nn.utils import clip_grad_norm_
import nltk
import string
from nltk.stem import WordNetLemmatizer
import sklearn.metrics as metrics
import pickle
import torchtext



# In[2]:


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# In[3]:
# In[4]:
# In[5]:
# In[6]:




'''
Special_Tokens=set(["<PAD>","<SOS>","<EOS>","<UNK>"])
Special_chars=set(["<pad>","<soc>","<eoc>"])

'''

class Word_Token:

    def __init__(self,token):

        self.token=token
        self.lemma=None
        self.token_map={}
        self.chars=[e for e in token]
        self.create_features()


    def __str__(self):
        
        M={}
        M["token_map"]=self.token_map
        M["char_list"]=self.chars
        return str(M)



    def create_features(self):

        #is_punctuation
        #is_numeric
        #is_all_lower_string,is_all_upper_string,only_first_upper_string,other_cases(like mRNA)
        #character_tokens
        #lemmatized_token and token are different

        token=self.token
        Special_Tokens=set(["<PAD>","<SOS>","<EOS>","<UNK>"])

        if token in Special_Tokens:

            self.token_map["is_lemma"]=0
            self.token_map["is_punc"]=0
            self.token_map["is_num"]=0
            self.token_map["is_lower"]=0


        else:

            lemmatizer=WordNetLemmatizer()
            lemma=lemmatizer.lemmatize(token)
            self.lemma=lemma.lower()

            if(token==lemma):
                self.token_map["is_lemma"]=1
            else:
                self.token_map["is_lemma"]=-1

            punctuations=set(string.punctuation)

            if(token in punctuations):
                self.token_map["is_punc"]=1
            else:
                self.token_map["is_punc"]=-1


            if(token.isnumeric()):
                self.token_map["is_num"]=1
            else:
                self.token_map["is_num"]=-1


            if(token.isalpha()):
                if(token.islower()):
                    self.token_map["is_lower"]=1

                elif(token.isupper()):
                    self.token_map["is_lower"]=-1

                else:
                    if(token[0].isupper()):
                        self.token_map["is_lower"]=-2
                    else:
                        self.token_map["is_lower"]=-1

            else:
                self.token_map["is_lower"]=0


# In[7]:
# In[8]:
def test_to_list(file_name):
    X=[]
    S=[]
    with open(file_name) as file:
        for line in file:
            l=line.rstrip()
            if(len(l)!=0):
                x=l.split()
                S.append(Word_Token(x[0]))                 
            else:
                X.append(S)
                S=[]
                
    if(len(S)!=0):
        X.append(S)
        S=[]
            
    return X

# In[9]:
# In[10]:


# def read_glove_vector(glove_path):    
#     with open(glove_path, 'r',encoding='UTF-8') as f:
#         words = set()
#         word_to_glove_map = {}
#         for line in f:
#             w_embed = line.split()
#             curr_word = w_embed[0]
#             word_to_glove_map[curr_word] = np.array(w_embed[1:], dtype=np.float32)
#         return word_to_glove_map


# In[11]:


# Glove_Map=read_glove_vector("glove.6B.300d.txt")
Glove_Map =torchtext.vocab.GloVe(name="6B",dim=300)


Types=["Chemical_Compound","Biological_Molecule", "Species"]
Tags=["B","I","E","S"]
Classes=["<PAD>","<START>","<END>"]


for e in Tags:
    for f in Types:
        Classes.append(e+"-"+f)

Classes+=["O"]


# In[12]:
# In[13]:
# In[14]:

path_to_test=sys.argv[1]
X_test=test_to_list(path_to_test)
# print([len(e) for e in X_test])
# print(len(X_test))
# print(X_test[0],X_test[1],X_test[2])



# In[15]:
# In[16]:
# In[17]:
# In[20]:




class NER_Dataset(Dataset):
    
    def __init__(self,X,embedder,embed_dim,unk_embed,text_to_index,class_to_index,char_to_index):
        self.inputs=X
        self.max_length=max([len(e) for e in X])
        self.text_to_index=text_to_index
        self.class_to_index=class_to_index
        self.embedder=embedder
        self.unk_embedding=unk_embed
        self.char_to_index=char_to_index
        

    def __len__(self):
        return len(self.inputs)
    
    

    def __getitem__(self, index):
        
        input_text   = self.inputs[index]     
        pad_value=self.text_to_index["<PAD>"]
        pad_label=self.class_to_index["<PAD>"]
        char_pad=self.char_to_index["<pad>"]
        
        
        S=np.full((self.max_length+2,1),pad_value,dtype='int64')
        E=np.zeros((self.max_length+2,embed_dim),dtype='float32')
        
        E_n=np.zeros((self.max_length+2,3),dtype='float32')
        E_c=np.zeros((self.max_length+2,5),dtype='float32')
        E_p=np.zeros((self.max_length+2,3),dtype='float32')
        E_l=np.zeros((self.max_length+2,3),dtype='float32')
        
        E_char=np.full((self.max_length+2,32),char_pad,dtype='int64')
      
        l=len(input_text)
        
        S[0][0]=self.text_to_index["<SOS>"]
        S[l+1][0]=self.text_to_index["<EOS>"]
        
        # T[0]=self.class_to_index["<START>"]
        # T[l+1]=self.class_to_index["<END>"]
        
              
        for i,word_token in enumerate(input_text):
            
            word=word_token.lemma
            
            if word in (self.embedder).stoi:
                E[i+1]=self.embedder[word].numpy()
            else:
                E[i+1]=self.unk_embedding
                
                
            if word in self.text_to_index:
                S[i+1][0]=self.text_to_index[word]
            else:
                S[i+1][0]=self.text_to_index["<UNK>"]
            
            E_n[word_token.token_map["is_num"]+1]=1
            E_c[word_token.token_map["is_lower"]+2]=1
            E_p[word_token.token_map["is_punc"]+1]=1
            E_l[word_token.token_map["is_lemma"]+1]=1
            
            l1=len(word_token.chars)
            
            if(l1>31):
                E_char[i+1][31]=self.char_to_index["<etc>"]                
            
            for j in range(0,min(31,l1)):
                char=word_token.chars[j]
                if char in self.char_to_index:
                    E_char[i+1][j]=self.char_to_index[char]
                else:
                    E_char[i+1][j]=self.char_to_index["<unk>"]
                
        
        # for i,word in enumerate(target_label):      
        #     T[i+1]=self.class_to_index[word]

        
        I=np.concatenate((E_n,E_c,E_p,E_l,E),axis=1)
        S=np.concatenate((S,E_char),axis=1)
        return torch.tensor(S),torch.tensor(I),torch.tensor(l)


# In[21]:


embedder=Glove_Map
embed_dim=300
feature_dim=14


with open('embedders', 'rb') as file:
    unk_embed,word_to_num,class_to_ind,char_to_num = pickle.load(file)


# In[22]:


batch_size = 512


test_dataset = NER_Dataset(X_test,embedder,embed_dim,unk_embed,word_to_num,class_to_ind,char_to_num)
test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)

# In[23]:


class BiLSTM(nn.Module):
    def __init__(self,vocab_size,char_size,num_classes,feature_dim,embed_dim,hidden_dim=256,char_dim=16,max_len=32):
        
        super(BiLSTM,self).__init__()
        
        
        self.feature_dim=feature_dim       
        self.word_embedding = nn.Embedding(vocab_size,hidden_dim,padding_idx=0)
        self.feature_embedding = nn.Linear(feature_dim,hidden_dim) 
        self.project=nn.Linear(embed_dim,hidden_dim)
        self.char_embedding=nn.Embedding(char_size,char_dim,padding_idx=0)
    
        
        self.lstm=nn.LSTM(hidden_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.char_lstm=nn.LSTM(max_len*char_dim,hidden_dim//2,batch_first=True,bidirectional=True)
        
        self.linear=nn.Linear(2*hidden_dim,num_classes)
                
        
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
                
        for param in self.char_lstm.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
                
                
        
      

        
    def forward(self,inp,embed): #input=(pretrained,features)       
        #directly load the embedding.
        
        
        inp_vec=inp[:,:,0]
        inp_vec=inp_vec.view(inp_vec.shape[0],inp_vec.shape[1])
        x_w=self.word_embedding(inp_vec)
        
        x_char_embed=self.char_embedding(inp[:,:,1:])
        x_char_embed_shape=x_char_embed.shape
        x_char_embed=x_char_embed.view(x_char_embed_shape[0],x_char_embed_shape[1],x_char_embed_shape[2]*x_char_embed_shape[3])
        
        x_f=self.feature_embedding(embed[:,:,:self.feature_dim])
        x_pt=self.project(embed[:,:,self.feature_dim:])
        
        
        
        x_char,_=self.char_lstm(x_char_embed)
        x=x_pt+x_f+x_w+x_char
        h_n,_=self.lstm(x)
        x=h_n       
        x=self.linear(x)
        output=x
        output=output.type(torch.float)
        
        return output


# In[24]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[25]:
# In[26]:


model=BiLSTM(len(word_to_num),len(char_to_num),len(Classes),feature_dim,embed_dim)
checkpoint=torch.load("cs1190369_model")
model.load_state_dict(checkpoint['model'])
model.to(device)

# In[27]:
# In[28]:
# In[29]:


def test(model,device,dataloader):

    model.eval()
    predicted_result=[]
    L=[]

    with torch.no_grad():

        for i,(src,emb,l) in enumerate(dataloader):           
            src,emb = src.to(device),emb.to(device) 
            output = model(src,emb)
            _, predicted = torch.max(output.data, 2)
            predicted_result+=list(predicted.cpu().numpy())
            L+=list(l.numpy())


    return predicted_result,L



# In[31]:
# In[32]:


def epoch_time(start_time,end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[33]:

Classes_decode=[e for e in Classes]
Classes_decode[0]="O"
Classes_decode[1]="O"
Classes_decode[2]="O"
possible_labels = ['B-Species', 'S-Species', 'S-Biological_Molecule', 'B-Chemical_Compound', 'B-Biological_Molecule', 'I-Species', 'I-Biological_Molecule', 'E-Species', 'E-Chemical_Compound', 'E-Biological_Molecule', 'I-Chemical_Compound', 'S-Chemical_Compound']


# In[34]:
# In[35]:



set_all_seeds(42)
file_name=sys.argv[2]
Predictions,Lengths=test(model,device,test_loader)
f=open(file_name,"w")
for i in range(0,len(Predictions)):
    for j in range(1,Lengths[i]+1):
        f.write(Classes_decode[Predictions[i][j]]+"\n")
    if(i!=len(Predictions)-1):
        f.write("\n")
