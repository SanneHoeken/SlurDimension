import csv, random
from tqdm import tqdm

def sample_contexts(data_path, contexts_path, encoded2keyword, sample_size, max_model_length=512):
    # Load data
    print("Loading data...")
    with open(data_path, 'r') as infile:
        reader = csv.DictReader(infile)
        data = [dic for dic in tqdm(reader)]

    print("Extracting contexts keyword mentions...")
    encoded_keywords = encoded2keyword.keys()

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


    # Write saved entries to csv-file
    with open(contexts_path, 'w') as outfile:
        header = keyword_entries[0].keys()
        writer = csv.DictWriter(outfile, fieldnames=header)
        writer.writeheader()
        for dic in keyword_entries:
            writer.writerow(dic)
