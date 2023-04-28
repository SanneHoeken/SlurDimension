from transformers import AutoTokenizer

def encode_queries(model_name, queries_file, type):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if type == 'keywords':
        # Get list of keywords
        with open(queries_file, 'r') as infile:
            keywords = [x.replace('\n', '') for x in infile.readlines()]

        # Encode keywords
        encoded_keywords = []
        for keyword in keywords:
            encoded_keywords.append(tokenizer.encode(keyword, add_special_tokens=False))

        # Write resulting keyword selection to txt file
        with open(queries_file.replace('.txt', '_encoded.txt'), 'w') as outfile:
            for word, encoded in zip(keywords, encoded_keywords):
                outfile.write(word+'\t'+str(encoded)[1:-1]+'\n')
    
    elif type == 'wordpairs':
        # Get list of wordpairs
        with open(queries_file, 'r') as infile:
            word_pairs = [tuple(x.replace('\n', '').split(';')) for x in infile.readlines()]

        # Encode wordpairs
        encoded_wordpairs = []
        for (word1, word2) in word_pairs:
            encoded1 = tokenizer.encode(word1, add_special_tokens=False)
            encoded2 = tokenizer.encode(word2, add_special_tokens=False)
            encoded_wordpairs.append((encoded1, encoded2))

        # Write resulting wordpair selection to txt file
        with open(queries_file.replace('.txt', '_encoded.txt'), 'w') as outfile:
            for (word1, word2), (encoded1, encoded2) in zip(word_pairs, encoded_wordpairs):
                outfile.write(word1+';'+word2+'\t'+str(encoded1)[1:-1]+';'+str(encoded2)[1:-1]+'\n')