import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=DeprecationWarning)


import torch
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW, T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from load_data import BARTDataset
import load_data
from tqdm import tqdm
import pickle
import argparse

def train(train_file_path, val_file_path, model_name):
    train_file_path = "/home/kshitiz/scratch/NLP/A3/data/train.jsonl"
    train_data = BARTDataset(train_file_path)
    val_file_path = "/home/kshitiz/scratch/NLP/A3/data/dev.jsonl"
    val_data = BARTDataset(val_file_path)

    # train_data = train_data+val_data

    batch_size = 8
    train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            collate_fn = load_data.collate_fn
        )
    val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            collate_fn = load_data.collate_fn
        )

    device = 'cuda:0'

    # Instantiate the BART model
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    model.resize_token_embeddings(len(train_data.bart_tokenizer))
    model = model.to(device)
    # Define the optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=1e-5)
    num_epochs = 20
    # warmup_steps = 5000
    total_steps = len(train_data) * num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=total_steps, anneal_strategy='linear', cycle_momentum=False)
    val_loss_best = 999
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].squeeze().to(device)
            attention_mask = batch['attention_mask'].squeeze().to(device)
            # decoder_input_ids = batch['decoder_input_ids'].squeeze().to(device)
            # decoder_attention_mask = batch['decoder_attention_mask'].squeeze().to(device)
            labels = batch['labels'].squeeze().to(device)
            # breakpoint()
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=labels)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Evaluate the model on the validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                input_ids = batch['input_ids'].squeeze().to(device)
                attention_mask = batch['attention_mask'].squeeze().to(device)
                # decoder_input_ids = batch['decoder_input_ids'].squeeze().to(device)
                # decoder_attention_mask = batch['decoder_attention_mask'].squeeze().to(device)
                labels = batch['labels'].squeeze().to(device)
                # outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=labels)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

        # Compute average training and validation loss
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f'Train loss: {train_loss:.4f} | Validation loss: {val_loss:.4f}')
        if(val_loss<val_loss_best):
            # model.save_pretrained(f'intent_in_input_epoch_{epoch}.pth')
            val_loss_best=val_loss
            model.save_pretrained(f'{model_name}')
 

from tqdm import tqdm

def test(test_data_file, out_file_path, model_dir):
    device = 'cuda:0'
    dataset = BARTDataset(test_data_file)
    data_loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            collate_fn = load_data.collate_fn
        )
    model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
    model = model.to(device)
    model.eval()
    outputs = []
    with open("t5_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with torch.no_grad():   
        for i,batch in enumerate(tqdm(data_loader)):
            # import pdb; pdb.set_trace()
            input_ids = batch['input_ids'].squeeze().to(device)
            generated_ids = model.generate(input_ids, num_beams=8, early_stopping=True, max_length = 1024)
            batch_text = []
            for j in range(generated_ids.shape[0]):
                generated_text = tokenizer.decode(generated_ids[j], skip_special_tokens=True)
                batch_text.append(generated_text)
            outputs+=batch_text
    with open(out_file_path, "w") as f:
        for i,text in enumerate(outputs):
            f.write(text+"\n")

def getArguments(parser):
    parser.add_argument("--train_file", help="training txt file")
    parser.add_argument("--test_file", help="testing txt file")
    parser.add_argument("--modelname", type=str, default="aiy227508_cs1190369_model", help="name of trained model")
    parser.add_argument("--out_file", type=str, default="outputfile.txt", help="name of output csv file")
    parser.add_argument('--val_file', help='val txt file')
    parser.add_argument('--mode', help='train or test')
    args = parser.parse_args()
    config = vars(args)
    return config

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="NLP A3", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config = getArguments(parser)

    modelname = config["modelname"]
    if(config["mode"]=="train"):
        train_data_path = config["train_file"] 
        val_data_path = config["val_file"]
        train(train_data_path, val_data_path, model_name=modelname)
    elif(config["mode"]=="test"):
        test_data_path = config["test_file"]
        out_file_path = config["out_file"]
        test(test_data_path, out_file_path, modelname)
    else:
        print("Unrecognized arguments")


# https://huggingface.co/docs/transformers/model_doc/bart