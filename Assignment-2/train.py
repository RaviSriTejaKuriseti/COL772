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
import sklearn
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


def train_to_list(file_name):
    X=[]
    Y=[]
    S=[]
    T=[]
    with open(file_name) as file:
        for line in file:
            l=line.rstrip()
            if(len(l)!=0):
                x,y=l.split()
                S.append(Word_Token(x))
                T.append(y)                    
            else:
                X.append(S)
                Y.append(T)
                S=[]
                T=[]
                
    if(len(S)!=0):
        X.append(S)
        Y.append(T)
        S=[]
        T=[]
            
    return X,Y



# In[9]:
# In[10]:


Glove_Map =torchtext.vocab.GloVe(name="6B",dim=300)



# In[12]:


Types=["Chemical_Compound","Biological_Molecule", "Species"]
Tags=["B","I","E","S"]
Classes=["<PAD>","<START>","<END>"]


for e in Tags:
    for f in Types:
        Classes.append(e+"-"+f)

Classes+=["O"]

class_to_ind={}
for i in range(0,len(Classes)):
    class_to_ind[Classes[i]]=i


# In[13]:
# In[14]:

path_to_train=sys.argv[1]
path_to_val=sys.argv[2]
X_train,Y_train=train_to_list(path_to_train)
X_val,Y_val=train_to_list(path_to_val)



# In[15]:


word_to_num={}
word_to_num["<PAD>"]=0
word_to_num["<UNK>"]=1
word_to_num["<SOS>"]=2
word_to_num["<EOS>"]=3
word_id=4


# In[16]:


char_to_num={}
char_to_num["<pad>"]=0
char_to_num["<unk>"]=1
char_to_num["<etc>"]=2
char_id=3


# In[17]:


ctr={}
class_ctr=np.array([0]*len(Classes))
class_ctr[0]=1
class_ctr[1]=len(X_train)
class_ctr[2]=class_ctr[1]

ct=0
for i in range(0,len(X_train)):
    for j in range(0,len(X_train[i])):            
        e=X_train[i][j].lemma
        key=class_to_ind[Y_train[i][j]]
        class_ctr[key]+=1
        if(Y_train[i][j]=="O"):
            if e not in ctr:
                ctr[e]=1
            else:
                ctr[e]+=1
        else:
            if e not in word_to_num:                
                word_to_num[e]=word_id
                word_id+=1
                
        for u in X_train[i][j].chars:
            if u not in char_to_num:
                char_to_num[u]=char_id
                char_id+=1
            


# In[18]:

set_all_seeds(42)
Word_frequencies=[(ctr[e],e) for e in ctr]
Word_frequencies.sort()
Word_frequencies.reverse()


# In[19]:


j=0
while(word_id<25000 and j<len(Word_frequencies)):
    if(Word_frequencies[j][1] not in word_to_num):
        e=Word_frequencies[j][1]
        word_to_num[e]=word_id
        word_id+=1
    j+=1


# In[20]:



class NER_Dataset(Dataset):
    
    def __init__(self,X,Y,embedder,embed_dim,unk_embed,text_to_index,class_to_index,char_to_index):
        self.inputs=X
        self.outputs=Y
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
        target_label = self.outputs[index]        
        pad_value=self.text_to_index["<PAD>"]
        pad_label=self.class_to_index["<PAD>"]
        char_pad=self.char_to_index["<pad>"]
        
        
        S=np.full((self.max_length+2,1),pad_value,dtype='int64')
        T=np.full(self.max_length+2,pad_label,dtype='int64')
        E=np.zeros((self.max_length+2,embed_dim),dtype='float32')
        
        E_n=np.zeros((self.max_length+2,3),dtype='float32')
        E_c=np.zeros((self.max_length+2,5),dtype='float32')
        E_p=np.zeros((self.max_length+2,3),dtype='float32')
        E_l=np.zeros((self.max_length+2,3),dtype='float32')
        
        E_char=np.full((self.max_length+2,32),char_pad,dtype='int64')
      
        l=len(input_text)
        
        S[0][0]=self.text_to_index["<SOS>"]
        S[l+1][0]=self.text_to_index["<EOS>"]
        
        T[0]=self.class_to_index["<START>"]
        T[l+1]=self.class_to_index["<END>"]
        
              
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
                
        
        for i,word in enumerate(target_label):      
            T[i+1]=self.class_to_index[word]

        
        I=np.concatenate((E_n,E_c,E_p,E_l,E),axis=1)
        S=np.concatenate((S,E_char),axis=1)
        return torch.tensor(S),torch.tensor(I),torch.tensor(T),torch.tensor(l)


# In[21]:


embedder=Glove_Map
embed_dim=300
unk_embed=np.random.normal((embed_dim,))
feature_dim=14


# In[22]:


batch_size = 512

train_dataset = NER_Dataset(X_train,Y_train,embedder,embed_dim,unk_embed,word_to_num,class_to_ind,char_to_num)
val_dataset   = NER_Dataset(X_val,Y_val,embedder,embed_dim,unk_embed,word_to_num,class_to_ind,char_to_num)

train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=False)
val_loader   = DataLoader(val_dataset, batch_size=batch_size,shuffle=False)


# In[23]:
with open('embedders','wb') as file:
    pickle.dump((unk_embed,word_to_num,class_to_ind,char_to_num),file)


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


loss_weights=[1.0]*len(Classes)
loss_weights[class_to_ind["<START>"]]=0
loss_weights[class_to_ind["<END>"]]=0
loss_weights[class_to_ind["<PAD>"]]=0

loss_weights[class_to_ind["O"]]=(1.0/999.0)
loss_weights=torch.tensor(loss_weights).type(torch.float).to(device)


# In[26]:


model=BiLSTM(len(word_to_num),len(char_to_num),len(Classes),feature_dim,embed_dim).to(device)
criterion = nn.CrossEntropyLoss(reduction='sum',weight=loss_weights)
optimizer = torch.optim.Adam(model.parameters(),lr=3e-3)


# In[27]:
# In[28]:


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# print(model)
# print(f'The model has {count_parameters(model):,} trainable parameters')


# In[29]:


def train(model,device,dataloader,optimizer,criterion,clip):

    model.train()
    epoch_loss = 0
    total_tokens=torch.tensor([0]).to(device)

    for _, (src,emb,tar,l) in enumerate(dataloader):
        src,emb,tar,l = src.to(device),emb.to(device),tar.to(device),l.to(device)
        optimizer.zero_grad()  
        output = model(src,emb)
        (a,b,c)=output.shape
        output=output.view(a*b,c)
        tar=tar.view(a*b)
        loss = criterion(output,tar)
        loss.backward()
        clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        total_tokens+=torch.sum(l)
        
    total_tokens=list(total_tokens.cpu().numpy())[0]
    mean_scaled_loss=(epoch_loss) / total_tokens

        
    return mean_scaled_loss


# In[30]:


def validate(model,device,dataloader,criterion):
    model.eval()
    epoch_loss = 0
    total_tokens=torch.tensor([0]).to(device)
    predicted_result=[]
    L=[]
    gold_result=[]
    
    with torch.no_grad():      
        for _, (src,emb,tar,l) in enumerate(dataloader):
            
            L+=list(l.numpy())
            src,emb,tar,l = src.to(device),emb.to(device), tar.to(device),l.to(device)         
            output = model(src,emb)
            
            _, predicted = torch.max(output.data, 2)
            predicted_result+=list(predicted.cpu().numpy())
            gold_result+=list(tar.cpu().numpy())
            
            
            
            (a,b,c)=output.shape
            output=output.view(a*b,c)
            tar=tar.view(a*b)
            loss = criterion(output,tar)
            epoch_loss += loss.item()
            total_tokens+=torch.sum(l)
            
        total_tokens=list(total_tokens.cpu().numpy())[0]
        mean_scaled_loss=(epoch_loss) / total_tokens
    
    return mean_scaled_loss,predicted_result,gold_result,L


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






N_EPOCHS = 100
CLIP = 1
output_dir=""

training_loss=[]
validation_loss=[]

best_valid_loss = float('inf')
best_score=0
best_weights=None

for epoch in tqdm(range(N_EPOCHS)):

    start_time = time.time()
    train_loss = train(model,device,train_loader,optimizer,criterion,CLIP)
    valid_loss,Pr,Go,Lens = validate(model,device,val_loader,criterion)
    
    training_loss.append(train_loss)
    validation_loss.append(valid_loss)
   
    pred_data=[]
    gold_data=[]

    i=0
    for i in range(0,len(Pr)):
        A=[]
        B=[]
        for j in range(1,Lens[i]+1):
            A.append(Classes_decode[Pr[i][j]])
            B.append(Classes_decode[Go[i][j]])
        pred_data+=A
        gold_data+=B

    f1_micro = metrics.f1_score(gold_data, pred_data, average="micro", labels=possible_labels)
    f1_macro = metrics.f1_score(gold_data, pred_data, average="macro", labels=possible_labels)
    score=0.5*(f1_micro+f1_macro)
    
    # print(f"f1_macro : {round(100*f1_macro,5)}")
    # print(f"f1_micro : {round(100*f1_micro,5)}")
    

    weights = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
    if(score>best_score):
        best_score=score
        best_weights=weights
 
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    # print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    # print(f'\tTrain Loss: {train_loss:.3f}')
    # print(f'\t Val. Loss: {valid_loss:.3f}')





torch.save(best_weights,"cs1190369_model")
 
    
    

