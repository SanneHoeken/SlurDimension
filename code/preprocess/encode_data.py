import csv, re
from transformers import AutoTokenizer
from tqdm import tqdm

def preprocess(text, lower, reddit):

    if lower:
        text = text.lower()
    if reddit:
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
    text = re.sub(r"https?:\/\/[^\s]+", '', text) #remove urls

    return text

def encode_data(model_name, input_path, lower=True, reddit=False):
    
    # Load input_data
    with open(input_path, 'r') as infile:
        reader = csv.DictReader(infile)
        data = [dic for dic in reader]

    # Load tokenizer and preprocess and encode posts
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for dic in tqdm(data):
        post = preprocess(dic['text'], lower, reddit)
        encoded = tokenizer.encode(post)
        dic['encoded'] = str(encoded)[1:-1]
    
    # Store encoding data in addition to original data
    with open(input_path.replace('.csv', '_encoded.csv'), 'w') as outfile:
       header = data[0].keys()
       writer = csv.DictWriter(outfile, fieldnames=header)
       writer.writeheader()
       for dic in data:
            writer.writerow(dic)