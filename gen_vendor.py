import pandas as pd          # import pandas for data read and preprocessing
import numpy as np           # import numpy for numerical python
import time

import torch                 #import torch for
from torch.utils.data import Dataset, DataLoader         # import dataset and dataloader use pre-loaded datasets
#from transformers import BartForConditionalGeneration, BartTokenizer,AdamW      #import from transformer to load model
from transformers import T5ForConditionalGeneration, T5Tokenizer




def g_vendor(description, max_length=200):
    
    
    st=time.time()
    name_model.eval()   #evaluate vname model
    type_model.eval()   #evaluate type model
    
    
   # Encode our descriprions via tokenizers
    inputs_name = name_tokenizer.encode_plus(
        description,
        max_length=200,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    
    name_input_ids = inputs_name["input_ids"].to(device)
    name_attention_mask = inputs_name["attention_mask"].to(device)
    
    # gradient=0 because Perform inference & memory utilize
    with torch.no_grad():
        name_outputs = name_model.generate(
            input_ids=name_input_ids,
            attention_mask=name_attention_mask,
            max_length=max_length,
            return_dict_in_generate=True,
            output_scores=True
        )

    # Extract logits and probabilities
    logits = name_outputs.sequences
    prob_name = torch.nn.functional.softmax(name_outputs.scores[-1], dim=-1)
    
    generated_vname = name_tokenizer.decode(logits[0], skip_special_tokens=True)

    # Log top probabilities and corresponding tokens for debugging
    top_prob_name, top_token_name = torch.topk(prob_name[0], 3)
    top_prob_name = top_prob_name.cpu().numpy()
    top_token_name = top_token_name.cpu().numpy()
    
    #top_text = [tokenizer.decode([token]) for token in top_token]
    
    # Convert the maximum probability to percentage
    
    prob_name = top_prob_name[0]* 100
    prob_name=round(prob_name,2)
    
    
    # For Type
    
    inputs_type = type_tokenizer.encode_plus(
        description,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    
    
    
    type_input_ids = inputs_type["input_ids"].to(device)
    type_attention_mask = inputs_type["attention_mask"].to(device)

    with torch.no_grad():
        type_outputs = type_model.generate(
            input_ids=type_input_ids,
            attention_mask=type_attention_mask,
            max_length=max_length,
            return_dict_in_generate=True,
            output_scores=True
        )

    # Extract logits and probabilities
    t_logits = type_outputs.sequences
    t_prob = torch.nn.functional.softmax(type_outputs.scores[-1], dim=-1)
    

    generated_type= type_tokenizer.decode(t_logits[0], skip_special_tokens=True)

    # Log top probabilities and corresponding tokens for debugging
    top_prob_type, top_token_type = torch.topk(t_prob[0], 3)
    top_prob_type = top_prob_type.cpu().numpy()
    prob_type=top_prob_type[0]*100
    prob_type= round(prob_type,2)
    
    top_token_type = top_token_type.cpu().numpy()
    #print(top_token)
    
    
    et = time.time()
    print("Inference time: ", et-st)

    
    
    
    
    
    return generated_vname, prob_name, generated_type, prob_type







def input_data(data):

# Vendor Name LLM And Type LLM load & Tokenized 

    vnmane_llm="t5_medium_llm"
    global name_model
    name_model = T5ForConditionalGeneration.from_pretrained(vnmane_llm)
    
    global name_tokenizer
    name_tokenizer = T5Tokenizer.from_pretrained(vnmane_llm)
    
    # Vendor Name Type LLM load & Tokenized 
    vtype_llm="t5_medium_llm_type"
    global type_model
    type_model = T5ForConditionalGeneration.from_pretrained(vtype_llm)
    
    global type_tokenizer
    type_tokenizer = T5Tokenizer.from_pretrained(vtype_llm)
    
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    name_model.to(device)
    type_model.to(device)
    print("Models and tokenizers loaded successfully")
    
    
    #data['Filtered_Descriptions'], data['gen_vendor_name'], data['name %'], data['generated_type'], data['prob_type %'] = zip(*data['Descriptions'].apply(g_vendor))

    name=[]
    namee=[]
    type=[]
    typee=[]
    for i in data:
        values=g_vendor(i)
        name.append(values[0])
        namee.append(values[1])
        type.append(values[2])
        typee.append(values[3])
        
    generated_list=list(zip(name,namee,type,typee))    
    

            
        
    
    return generated_list
    




#text=['ALLSTATE INS CO ORIG ID DESC DATE MAY CO ENTRY DESCR INS PREM SEC PPD TRACE EED IND ID IND NAME KLEMM TRN TC','TESLA MOTORS ORIG ID DESC DATE B CO ENTRY DESCR TESLA MOTOSEC PPD TRACE EED IND ID IND NAME MATTHEW KLEMM TRN TC',
#    'NON CHASE ATM FEE WITH']
#
#
#
#data=pd.DataFrame({
#    'text':text
#})
#
#
###description = "aplpay gelateria gennew york ny"
#data1=data['text'].apply(generate_vendor)
#print(data1)
##
#
#
#