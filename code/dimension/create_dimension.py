import pickle, torch, random

def create_dimension(input_path, output_path, wordpairs_file, method, sample=None):

    # Get list of wordpairs
    with open(wordpairs_file, 'r') as infile:
        word_pairs = [tuple(x.replace('\n', '').split(';')) for x in infile.readlines()]

    with open(input_path, 'rb') as infile:
        embeddings = pickle.load(infile)

    mean_vec1s = []
    mean_vec2s = []
    diff_vecs = []

    # average over context representations for every word
    # calculate word pair difference based on mean representations 
    for (word1, word2) in word_pairs:
        word1_vecs = embeddings[word1]['embeddings']
        word2_vecs = embeddings[word2]['embeddings']
        if sample:
            word1_vecs = random.sample(word1_vecs, sample)
            word2_vecs = random.sample(word2_vecs, sample)
        mean_vec1 = torch.mean(torch.stack(word1_vecs), 0)
        mean_vec2 = torch.mean(torch.stack(word2_vecs), 0)
        diff_vec = mean_vec1 - mean_vec2

        mean_vec1s.append(mean_vec1)
        mean_vec2s.append(mean_vec2)
        diff_vecs.append(diff_vec)
            
    if method == 'kozlowski':
        # take mean of difference vectors
        dimension = torch.mean(torch.stack(diff_vecs), 0)
        #print(dimension)
    elif method == 'garg':
        # not correct implemented yet?
        agg1 = torch.mean(torch.stack(mean_vec1s), 0)
        agg2 = torch.mean(torch.stack(mean_vec2s), 0)
        dimension = agg1 - agg2

    # write dimension embedding to file
    with open(output_path, 'wb') as outfile:
        pickle.dump({'dimension': dimension}, outfile)
    