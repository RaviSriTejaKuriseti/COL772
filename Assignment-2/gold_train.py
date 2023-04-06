
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
                S.append(x)
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

X_train,Y_train=train_to_list("train1.txt")
f=open("gold_train.txt","w")
for i in range(0,len(Y_train)):
    # print(Lengths[i])
    for j in range(0,len(Y_train[i])):
        f.write(Y_train[i][j]+"\n")
    if(i!=len(Y_train)-1):
        f.write("\n")