import pickle, torch, csv

def dimension_projection(input_path, dimension_path, output_path):
    
    cos = torch.nn.CosineSimilarity(dim=0)

    with open(input_path, 'rb') as infile:
        embeddings = pickle.load(infile)

    with open(dimension_path, 'rb') as infile:
        dimension_dic = pickle.load(infile)
        dimension = dimension_dic['dimension']

    output = []
    for test_word in embeddings:
        word_vecs = embeddings[test_word]['embeddings']
        post_ids = embeddings[test_word]['post_ids']
        for vec, post_id in zip(word_vecs, post_ids):
            cossim = cos(vec, dimension).item()
            output_dic = {'LU': test_word, 'post_id': post_id, 'projection': cossim}
            output.append(output_dic)

    with open(output_path, 'w') as outfile:
        header = output[0].keys()
        writer = csv.DictWriter(outfile, fieldnames=header)
        writer.writeheader()
        for dic in output:
            writer.writerow(dic)
