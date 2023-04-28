import csv, random
from tqdm import tqdm

def sample_contexts(data_path, contexts_path, queries_file, type, sample_size, max_model_length=512):
    # Load data
    print("Loading data...")
    with open(data_path, 'r') as infile:
        reader = csv.DictReader(infile)
        data = [dic for dic in tqdm(reader)]

    if type == 'keywords':
        print("Extracting contexts keyword mentions...")
        # Get list of keywords
        with open(queries_file, 'r') as infile:
            lines = [x.replace('\n', '') for x in infile.readlines()]    
        keywords = [line.split('\t')[0] for line in lines]
        encoded_keywords = [line.split('\t')[1] for line in lines]
        encoded2keyword = {e: k for e, k in zip(encoded_keywords, keywords)}

        # Map keywords to the entries in wich they occur
        key2entry = {key: list() for key in encoded_keywords}
        for entry in tqdm(data):
            encoded = entry['encoded']
            if len(encoded.split(', ')) <= max_model_length:
                for key in encoded_keywords:
                    if ' '+key+',' in encoded: 
                        key2entry[key].append(entry.copy())

        # Save sample of keyword-mentioning entries with specified size to list
        keyword_entries = []
        for key, entries in key2entry.items():
            keyword = encoded2keyword[key]
            for entry in entries:
                entry['keyword'] = keyword
                entry['keyword_encoded'] = key
            sample = entries
            if sample_size:    
                sample = random.sample(entries, sample_size)
            keyword_entries.extend(sample)

    
    elif type == 'wordpairs':
        print("Extracting contexts with keyword mentions...")
        # Get list of wordpairs
        with open(queries_file, 'r') as infile:
            lines = [x.replace('\n', '') for x in infile.readlines()] 
        word_pairs = [tuple(line.split('\t')[0].split(';')) for line in lines] 
        encoded_wordpairs = [tuple(line.split('\t')[1].split(';')) for line in lines]

        # Map keywords to the entries in wich they occur
        all_encodings = [item for t in encoded_wordpairs for item in t]
        key2entry = {key: list() for key in all_encodings}
        for entry in tqdm(data):
            encoded = entry['encoded']
            if len(encoded.split(', ')) <= max_model_length:
                for key in all_encodings:
                    if ' '+key+',' in encoded: 
                        key2entry[key].append(entry.copy())
        
        # Save sample of word-mentioning entries with specified size to list
        # if both words of word pairs are mentioned in at least 1 entry
        keyword_entries = []
        for (w1, w2), (w1_encoding, w2_encoding) in zip(word_pairs, encoded_wordpairs):

            w1_sample = key2entry[w1_encoding]
            for w1_entry in w1_sample:
                w1_entry['keyword'] = w1
                w1_entry['keyword_encoded'] = w1_encoding
            
            w2_sample = key2entry[w2_encoding]
            for w2_entry in w2_sample:
                w2_entry['keyword'] = w2
                w2_entry['keyword_encoded'] = w2_encoding
            
            for sample in [w1_sample, w2_sample]:
                if sample_size:
                    sample = random.sample(sample, sample_size)
                keyword_entries.extend(sample)
    
    # Write saved entries to csv-file
    with open(contexts_path, 'w') as outfile:
        header = keyword_entries[0].keys()
        writer = csv.DictWriter(outfile, fieldnames=header)
        writer.writeheader()
        for dic in keyword_entries:
            writer.writerow(dic)
