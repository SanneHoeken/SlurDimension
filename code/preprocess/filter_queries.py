import csv
from tqdm import tqdm

def filter_queries(data_path, queries_file, output_file, type, thres, max_model_length=512):
    # Load data
    print("Loading data...")
    with open(data_path, 'r') as infile:
        reader = csv.DictReader(infile)
        data = [dic for dic in tqdm(reader)]

    if type == 'keywords':
        print("Counting keyword mentions in contexts...")
        # Get list of keywords
        with open(queries_file, 'r') as infile:
            lines = [x.replace('\n', '') for x in infile.readlines()]    
        keywords = [line.split('\t')[0] for line in lines]
        encoded_keywords = [line.split('\t')[1] for line in lines]
        encoded2keyword = {e: k for e, k in zip(encoded_keywords, keywords)}

        # Map keywords to occuring frequency
        key2freq = {key: 0 for key in encoded_keywords}
        for entry in tqdm(data):
            encoded = entry['encoded']
            if len(encoded.split(', ')) <= max_model_length:
                for key in encoded_keywords:
                    if ' '+key+',' in encoded: 
                        key2freq[key] += 1

        # Filter keywords with frequency above threshold
        resulting_keywords = []
        for key, freq in key2freq.items():
            keyword = encoded2keyword[key]
            if freq >= thres: 
                resulting_keywords.append(keyword+'\t'+key)
                print(keyword, '\t', freq)

        # Write resulting keyword selection to txt file
        with open(queries_file.replace('.txt', '_selection.txt'), 'w') as outfile:
            for result in resulting_keywords:
                outfile.write(result+'\n')
    
    elif type == 'wordpairs':
        print("Counting keyword mentions in contexts...")
        # Get list of wordpairs
        with open(queries_file, 'r') as infile:
            lines = [x.replace('\n', '') for x in infile.readlines()] 
        word_pairs = [tuple(line.split('\t')[0].split(';')) for line in lines] 
        encoded_wordpairs = [tuple(line.split('\t')[1].split(';')) for line in lines]

        # Map keywords to occuring frequencies
        all_encodings = [item for t in encoded_wordpairs for item in t]
        key2freq = {key: 0 for key in all_encodings}
        for entry in tqdm(data):
            encoded = entry['encoded']
            if len(encoded.split(', ')) <= max_model_length:
                for key in all_encodings:
                    if ' '+key+',' in encoded: 
                        key2freq[key] += 1
        
        # Filter wordpairs with frequencies above threshold
        resulting_wordpairs = []
        
        for (w1, w2), (w1_encoding, w2_encoding) in zip(word_pairs, encoded_wordpairs):

            w1_freq = key2freq[w1_encoding]
            w2_freq = key2freq[w2_encoding]

            if w1_freq >= thres and w2_freq >= thres: 
                resulting_wordpairs.append(w1+';'+w2+'\t'+w1_encoding+';'+w2_encoding)
                print(w1, '\t', w1_freq, '\t\t', w2, '\t', w2_freq)
        
        # Write resulting wordpair selection to txt file
        with open(output_file, 'w') as outfile:
            for result in resulting_wordpairs:
                outfile.write(result+'\n')

    

if __name__ == '__main__':

    thres = 10
    data_path = "" 
    queries_file = "" 
    output_file = ""
    type = 'keywords' # or 'wordpairs'
    filter_queries(data_path, queries_file, output_file, type, thres)   