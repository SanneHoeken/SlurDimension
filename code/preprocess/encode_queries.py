from transformers import AutoTokenizer

def encode_queries(model_name, queries_file, type):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if type == 'keywords':
        # Get list of keywords
        with open(queries_file, 'r') as infile:
            keywords = [x.replace('\n', '') for x in infile.readlines()]

    elif type == 'wordpairs':
        # Get list of wordpairs
        with open(queries_file, 'r') as infile:
            word_pairs = [tuple(x.replace('\n', '').split(';')) for x in infile.readlines()]
        keywords = list(set([item for t in word_pairs for item in t]))

    # Encode keywords
    encoded_keywords = []
    for keyword in keywords:
        encoded_keywords.append(tokenizer.encode(keyword, add_special_tokens=False))

    encoded2keyword = {str(e)[1:-1]: k for e, k in zip(encoded_keywords, keywords)}
    
    return encoded2keyword