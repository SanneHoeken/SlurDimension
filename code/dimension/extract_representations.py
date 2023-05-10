import torch, pickle, csv
from transformers import AutoModel
from tqdm import tqdm

def extract_representations(model_name, input_path, output_path, layer_selection, layer_aggregation):

    print('Extract representations...')
    # Load input_data
    with open(input_path, 'r') as infile:
        reader = csv.DictReader(infile)
        data = [dic for dic in reader]

    keywords = list(set([post['keyword'] for post in data]))
    
    # Load model
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    
    keywords = list(set([post['keyword'] for post in data]))
    embeddings = {w : {'embeddings': [], 'post_ids': []} for w in keywords}

    # Iterate over posts and collect representations
    for post in tqdm(data):
        
        post_id = post['id']
        keyword = post['keyword']
        subwords_ids = [int(i) for i in post['encoded'].split(', ')]
        keyword_ids = [int(i) for i in post['keyword_encoded'].split(', ')]

        # Get indexes of keyword ids in encoded post
        found = False
        for i in range(len(subwords_ids)):
            if not found:
                if subwords_ids[i:i+len(keyword_ids)] == keyword_ids:
                    start_idx = i
                    end_idx = i+len(keyword_ids)
                    found = True

        # feed post text to the model    
        input_ids = torch.tensor([subwords_ids])
        encoded_layers = model(input_ids)[-1]
        
        # extract (aggregated) selection of hidden layer(s)
        if type(layer_selection) == int:
            vecs = encoded_layers[layer_selection].squeeze(0)
        elif type(layer_selection) == list:
            selected_encoded_layers = [encoded_layers[x] for x in layer_selection]
            if layer_aggregation == 'mean':
                vecs = torch.mean(torch.stack(selected_encoded_layers), 0).squeeze(0)
        elif layer_selection == 'all':
            if layer_aggregation == 'mean':
                vecs = torch.mean(torch.stack(encoded_layers), 0).squeeze(0)
        
        # target word selection 
        vecs = vecs.detach()
        vecs = vecs[start_idx:end_idx]
        # aggregate sub-word embeddings (by averaging)
        vector = torch.mean(vecs, 0)
        
        embeddings[keyword]['embeddings'].append(vector)
        embeddings[keyword]['post_ids'].append(post_id)
    
    with open(output_path, 'wb') as outfile:
        pickle.dump(embeddings, outfile)