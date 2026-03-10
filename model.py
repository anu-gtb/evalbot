## Importing libraries
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import torch
#from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import BertTokenizer,BertForSequenceClassification

## Loading BERT Tokenizer and Model
model_name='aryashukla/evalbot'  
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
model=BertForSequenceClassification.from_pretrained(model_name)

## Text Cleaning methods
def remove_special_characters(text):
    translator=str.maketrans('','',string.punctuation)
    return text.translate(translator)
 
def lower_case(text):
    text=text.lower()
    return text

def remove_stopwords(text):
    words=text.split()
    stop_words=set(stopwords.words('english'))
    words=[word for word in words if word not in stop_words]
    return ' '.join(words)

## Tokenization of cleaned text data
def tokenization(sent1,sent2):
    encoded=tokenizer.encode_plus(
        sent1,sent2,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    input_ids=encoded['input_ids'] 
    attention_masks=encoded['attention_mask'] 
    return input_ids,attention_masks

## Applying text cleaning functions
def preprocess(data):
    data[data.columns[0]]=data[data.columns[0]].apply(remove_special_characters)
    data[data.columns[0]]=data[data.columns[0]].apply(lower_case)
    data[data.columns[0]]=data[data.columns[0]].apply(remove_stopwords)
    data[data.columns[1]]=data[data.columns[1]].apply(remove_special_characters)
    data[data.columns[1]]=data[data.columns[1]].apply(lower_case)
    data[data.columns[1]]=data[data.columns[1]].apply(remove_stopwords)
    data=data.reset_index(drop=True)
    pairs=[(row[data.columns[0]],row[data.columns[1]]) for index,row in data.iterrows()]
    return pairs

## Prediction of marks and accuracies/similarities
def predict_score_and_accuracy(data):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    marks_obtained=[]
    sent_pairs=preprocess(data)
    for sent1,sent2 in sent_pairs:
        input_ids,attention_mask=tokenization(sent1,sent2)
        with torch.no_grad():
            model.to(device)
            input_ids=input_ids.to(device)
            attention_masks=attention_mask.to(device)
            outputs=model(input_ids,attention_masks)
            logits=outputs.logits
            mark_obtained=torch.sigmoid(logits)
            accuracy=torch.argmax(logits,dim=1).item()
            if accuracy==0:
                avg_marks=torch.min(mark_obtained).item()
            elif accuracy==1:
                avg_marks=torch.mean(mark_obtained).item()
            elif accuracy==2:
                avg_marks=torch.max(mark_obtained).item()
        marks_obtained.append(avg_marks)
    return marks_obtained
