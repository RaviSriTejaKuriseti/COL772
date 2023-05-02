import json

def parse(tokens):
    if "(" not in tokens:
        assert ")" not in tokens
        ret = dict()
        start = 0
        mid = 0
        for ii, tok in enumerate(tokens):
            if tok == "«":
                mid = ii
            elif tok == "»":
                key = ' '.join(tokens[start:mid])
                val = ' '.join(tokens[mid + 1:ii])
                ret[key] = val
                start = mid = ii + 1
        return ret

    st = tokens.index("(")
    outer_key = ' '.join(tokens[0:st])
    assert tokens[-1] == ")", " ".join(tokens)

    level = 0
    last = st + 1
    ret = dict()
    for ii in range(st + 1, len(tokens) - 1, 1):
        tok = tokens[ii]
        if tok == "»" and level == 0:
            rr = parse(tokens[last:ii + 1])
            ret.update(rr)
            last = ii + 1
        elif tok == "(":
            level += 1
        elif tok == ")":
            level -= 1
            if level == 0:
                rr = parse(tokens[last:ii + 1])
                ret.update(rr)
                last = ii + 1

    return {outer_key: ret}



def read_jsonl_file(input_path):
    with open(input_path) as f:
        data = [json.loads(line) for line in f]
        return data
    
data=read_jsonl_file("data/dev.jsonl")
Golds=[e["output"] for e in data]
get_intent = lambda x: x.split('(', 1)[0].strip()


with open("data/t5_outputs_dev_32_19.txt") as f:
    Our_Output = [line.strip() for line in f]

Intent_Mismatch=[]
Parse_Mismatch=[]
Output_Mismatch=[]

for i,(gold,pred) in enumerate(zip(Golds,Our_Output)):
    if(gold!=pred):
        Output_Mismatch.append((i,gold,pred))

    try:
        _ = parse(pred.split())
    except:
        Parse_Mismatch.append((i,gold,pred))

    gintent = get_intent(gold)
    pintent = get_intent(pred)

    if(gintent!=pintent):
        Intent_Mismatch.append((i,gold,pred))


with open("error_analysis/t5_outputs_dev_1_5_intent_32_19.txt","w") as f:
    for e in Intent_Mismatch:
        f.write("Testcase-"+str(e[0])+"\n")
        f.write("Predicted: "+str(e[2])+"\n")
        f.write("Actual: "+str(e[1])+"\n")
        f.write("\n")

    
with open("error_analysis/t5_outputs_dev_1_5_parsing_32_19.txt","w") as f:
    for e in Parse_Mismatch:
        f.write("Testcase-"+str(e[0])+"\n")
        f.write("Predicted: "+str(e[2])+"\n")
        f.write("Actual: "+str(e[1])+"\n")
        f.write("\n")        

    

with open("error_analysis/t5_outputs_dev_1_5_output_32_19.txt","w") as f:
    for e in Output_Mismatch:
        f.write("Testcase-"+str(e[0])+"\n")
        f.write("Predicted: "+str(e[2])+"\n")
        f.write("Actual: "+str(e[1])+"\n")
        f.write("\n")  




