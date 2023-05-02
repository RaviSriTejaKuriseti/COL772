import json
from transformers import T5Tokenizer

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pickle

class BARTDataset(Dataset):
    def __init__(self, data_file):
        self.data = self.load_jsonl_file(data_file)

        self.input_sep_start, self.input_sep_end = '<INPUT_SEP_STRT>' , '<INPUT_SEP_END>'
        self.history_sep_start, self.history_sep_end = '<HISTORY_SEP_STRT>' , '<HISTORY_SEP_END>'
        self.list_sep_start, self.list_sep_end = '<LIST_SEP_STRT>' , '<LIST_SEP_END>'
        self.note_sep_start, self.note_sep_end = '<NOTE_SEP_STRT>' , '<NOTE_SEP_END>'
        self.contact_sep_start, self.contact_sep_end = '<CONTACT_SEP_STRT>' , '<CONTACT_SEP_END>'

        self.disfluency_token = "<disfluent>"

        self.user_query, self.response_text = "<user_query>", "<response_text>"
        self.name_user_list, self.item_user_list = "<name_user_list>", "<item_user_list>"
        self.name_user_notes, self.name_user_content= "<name_user_notes>", "<name_user_content>"

        # self.intent_sep_token_strt, self.intent_sep_token_end = '<INTENT_SEP_STRT>', '<INTENT_SEP_END>'
        # self.slot_sep_token_strt, self.slot_sep_token_end = '<SLOT_SEP_STRT>', '<SLOT_SEP_END>'
        # self.key_value_sep_token_strt, self.key_value_sep_token_end = '<KEY_VALUE_SEP_STRT>', '<KEY_VALUE_SEP_END>'
        # self.nested_sep_token_strt, self.nested_sep_token_end = '<NESTED_SEP_STRT>', '<NESTED_SEP_END>'

        self.sep_token='<SEP>'

        self.new_tokens = [self.input_sep_start, #self.input_sep_end, 
                           self.history_sep_start, #self.history_sep_end, 
                           self.list_sep_start, #self.list_sep_end,
                           self.note_sep_start, #self.note_sep_end, 
                           self.contact_sep_start, #self.contact_sep_end,
                           self.disfluency_token,
                           self.user_query, self.response_text,
                           self.name_user_list, self.item_user_list,
                           self.name_user_notes, self.name_user_content,
                        #    self.intent_sep_token_strt, self.intent_sep_token_end, self.slot_sep_token_strt, self.slot_sep_token_end,
                        #    self.key_value_sep_token_strt, self.key_value_sep_token_end, self.nested_sep_token_strt, self.nested_sep_token_end,
                           self.sep_token]
        
        # self.new_tokens=[]
        self.intent_tokens=[]
        for i in range(0,33):
            tok="<INTENT_TOK_{}>".format(i)
            self.intent_tokens.append(tok)

        self.new_tokens=self.new_tokens+self.intent_tokens
        self.intent_map={'Send_digital_object': 0, 'Get_health_stats': 1, 'Get_message_content': 2, 'Add_contact': 3, 'Initiate_call': 4, 'Create_note': 5, 'Add_item_to_list': 6, 'Create_list': 7, 'Get_list': 8, 'Order_menu_item': 9, 'Find_parking': 10, 'Get_note': 11, 'Start_exercise': 12, 'Stop_exercise': 13, 'Resume_exercise': 14, 'Pause_exercise': 15, 'Log_exercise': 16, 'Log_nutrition': 17, 'Check_order_status': 18, 'Get_bill': 19, 'Get_security_price': 20, 'Open_app': 21, 'Pay_bill': 22, 'Get_product': 23, 'Other': 24, 'Post_message': 25, 'Record_video': 26, 'Take_photo': 27, 'Cancel_ride': 28, 'Order_ride': 29, 'BuyEventTickets': 30, 'Play_game': 31, 'GetGenericBusinessType': 32}
        self.intents=[self.intent_map[e] for e in self.intent_map]

        self.bart_tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.bart_tokenizer.add_tokens(self.new_tokens)
        with open("t5_tokenizer.pkl", "wb") as f:
            pickle.dump(self.bart_tokenizer, f)
        # exit(0)

    
    def load_jsonl_file(self, filepath):
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def tokenize_and_encode_input(self, input_data):
        input_q, history, user_lists, user_notes, user_contacts = input_data["input"], input_data["history"], input_data["user_lists"], input_data["user_notes"], input_data["user_contacts"]       

        if("pattern" in input_data.keys() and input_data["pattern"]!=""):
            input_tokens = [self.disfluency_token]
        else:
            input_tokens = []

        get_intent = lambda x: x.split('(', 1)[0].strip()
        intent_value=self.intent_map[get_intent(input_data["output"])]

        # Tokenize input
        # input_tokens = []
        input_tokens += [self.input_sep_start]
        input_tokens += self.bart_tokenizer.tokenize(input_q)
        input_tokens+=[self.intent_tokens[intent_value]]
        # input_tokens += [self.input_sep_end]

        # Tokenize the history
        history_tokens = []
        history_tokens += [self.history_sep_start]
        for item in history:
            history_tokens += ["<user_query>"]
            history_tokens += self.bart_tokenizer.tokenize(item['user_query'])
            # history_tokens += [self.sep_token]
            history_tokens += ["<response_text>"]
            history_tokens += self.bart_tokenizer.tokenize(item['response_text'])
            # history_tokens += [self.sep_token]
        # history_tokens += [self.history_sep_end]
        
        # Tokenize the user lists
        user_lists_tokens = []
        user_lists_tokens += [self.list_sep_start]
        for item in user_lists:
            user_lists_tokens += ["<name_user_list>"]
            user_lists_tokens += self.bart_tokenizer.tokenize(item['name'])
            # user_lists_tokens += [self.sep_token]
            user_lists_tokens += ["<items_user_list>"]
            for list_item in item['items']:
                user_lists_tokens += self.bart_tokenizer.tokenize(list_item)
                user_lists_tokens += [self.sep_token]
        # user_lists_tokens += [self.list_sep_end]
        
        # Tokenize the user notes
        user_notes_tokens = []
        user_notes_tokens += [self.note_sep_start]
        for item in user_notes:
            user_notes_tokens += ["<name_user_notes>"]
            user_notes_tokens += self.bart_tokenizer.tokenize(item['name'])
            # user_notes_tokens += [self.sep_token]
            user_notes_tokens += ["<name_user_content>"]
            user_notes_tokens += self.bart_tokenizer.tokenize(item['content'])
            # user_notes_tokens += [self.sep_token]
        # user_notes_tokens += [self.note_sep_end]
        
        # Tokenize the user contacts
        user_contacts_tokens = []
        user_contacts_tokens += [self.contact_sep_start]
        for contact in user_contacts:
            user_contacts_tokens += self.bart_tokenizer.tokenize(contact)
            user_contacts_tokens += [self.sep_token]
        # user_contacts_tokens += [self.contact_sep_end]

        # Combine all the token lists with the separator token
        all_input_tokens = input_tokens + history_tokens + user_lists_tokens + user_notes_tokens + user_contacts_tokens
        # Encode the tokens into input IDs
        # all_input_tokens+=[]
        input_ids = self.bart_tokenizer.convert_tokens_to_ids(all_input_tokens)
        
        return input_ids


    def tokenize_and_encode_output(self,output_text):
        output_ids = self.bart_tokenizer.encode(output_text)
        return output_ids

    def create_mask(self, sample):
        mask_0 = sample.ne(0)
        mask_1 = sample.ne(1)
        return mask_0*mask_1
        

        
    def __getitem__(self, idx):
        # Get the input and output strings
        input_data = self.data[idx]
        # Create the input and output tensors
        # print(input_data)
        input_ids = self.tokenize_and_encode_input(input_data)
        # print(input_ids)
        # import pdb; pdb.set_trace()
        output_ids = self.tokenize_and_encode_output(input_data["output"])
        
        input_ids = torch.tensor(input_ids) #+ output_ids[1:])

        # Combine the input and output tensors with the separator token
        attention_mask = self.create_mask(input_ids)
        # decoder_attention_mask = self.create_mask(output_ids)
        
        # Return the input and output tensors
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            # 'decoder_input_ids': output_ids[:-1],
            # 'decoder_attention_mask': decoder_attention_mask[:-1],
            "labels" : torch.tensor(output_ids),
        }

def load_jsonl_file(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

import re
def collate_fn(examples):
    input_ids = [example['input_ids'].squeeze() for example in examples]
    attention_mask = [example['attention_mask'].squeeze() for example in examples]
    # decoder_input_ids = [example['decoder_input_ids'].squeeze() for example in examples]
    # decoder_attention_mask = [example['decoder_attention_mask'].squeeze() for example in examples]
    labels = [example['labels'].squeeze() for example in examples]
    
    # Pad the sequences to the same length within a batch
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=1)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=1)
    # decoder_input_ids = nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=1)
    # decoder_attention_mask = nn.utils.rnn.pad_sequence(decoder_attention_mask, batch_first=True, padding_value=1)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    # return {'input_ids': input_ids, 'attention_mask': attention_mask, 'decoder_input_ids': decoder_input_ids, 'decoder_attention_mask': decoder_attention_mask, 'labels': labels}
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


if __name__ == '__main__':
    train_file_path = "/home/kshitiz/scratch/NLP/A3/data/train.jsonl"
    data = load_jsonl_file(train_file_path)
    # import pdb; pdb.set_trace()
    # val_file_path = "/home/kshitiz/scratch/NLP/A3/data/dev.jsonl"

    # Create the BART dataset
    bart_dataset = BARTDataset(train_file_path)
    # Create the BART dataloader
    batch_size = 2
    bart_dataloader = DataLoader(
        bart_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True,
        collate_fn = collate_fn
    )
    for i, batch in enumerate(bart_dataloader):
        import pdb; pdb.set_trace()
        continue

