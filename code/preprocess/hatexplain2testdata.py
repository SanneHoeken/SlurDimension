import json, csv, nltk
from collections import defaultdict, Counter
from nltk.corpus import stopwords

def convert_data(dataset_file, data_outputfile, rationale_outputfile, freq_thres):
    
    stopwords = set(stopwords.words("english")) 
    
    with open(dataset_file, 'r') as infile:
        json_dict = json.load(infile)

    output_data = []
    token_rationale = []
    token2freq = defaultdict(int)

    for post_id, entry in json_dict.items(): 
        
        for token in entry['post_tokens']:
            token2freq[token] += 1
        
        added_tokens = []
        for i, token in enumerate(entry['post_tokens']):
            if token not in added_tokens:
                token_scores = [rationale[i] for rationale in entry['rationales']]
                majority_score = 0
                if token_scores:
                    majority_score = max(set(token_scores), key=token_scores.count)

                token_dic = {'LU': token,
                    'post_id': post_id,
                    'gold': majority_score}
                token_rationale.append(token_dic)
                added_tokens.append(token)

        labels = [a['label'] for a in entry['annotators']]
        majority_label = max(set(labels), key=labels.count)
        targets = [a['target'] for a in entry['annotators']]
        targets = Counter(sum(targets, []))
        majority_targets = [k for k,v in targets.items() if v == max(targets.values())]
        
        entry_dic = {'text': ' '.join(entry['post_tokens']),
                    'id': post_id,
                    'label': majority_label,
                    'target': majority_targets}
        output_data.append(entry_dic)

    with open(data_outputfile, 'w') as outfile:
        header = output_data[0].keys()
        writer = csv.DictWriter(outfile, fieldnames=header)
        writer.writeheader()
        for dic in output_data:
            writer.writerow(dic)


    noun_rationale = []
    for dic in token_rationale:
        # filter for nouns, with at least one letter, 
        # that are not stopwords, and occur more than frequency threshold
        token = dic['LU']
        tag = nltk.pos_tag([token])[0][1] 
        freq = token2freq[token]
        if all([freq >= freq_thres,
            any(letter.isalpha() for letter in token),
            tag.startswith("NN"),
            token not in stopwords]):
            
            noun_rationale.append(dic)

    with open(rationale_outputfile, 'w') as outfile:
        header = noun_rationale[0].keys()
        writer = csv.DictWriter(outfile, fieldnames=header)
        writer.writeheader()
        for dic in noun_rationale:
            writer.writerow(dic)


if __name__ == '__main__':

    freq_thres = 10
    dataset_file = '../../../Data/HateXplain/HateXplain.json'
    rationale_outputfile = '../../data/hatexplain/HateXplain_nouns_rationales.csv'
    data_outputfile = '../../data/hatexplain/HateXplain_data_modified.csv'

    convert_data(dataset_file, data_outputfile, rationale_outputfile, freq_thres)