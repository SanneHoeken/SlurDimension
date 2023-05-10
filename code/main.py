import os
from preprocess.encode_data import encode_data
from preprocess.encode_queries import encode_queries
from dimension.sample_contexts import sample_contexts
from dimension.extract_representations import extract_representations
from dimension.create_dimension import create_dimension
from dimension.dimension_projection import dimension_projection

def main(data_path, lu_path, contexts_path, dimension_path, representations_path, 
             projections_path, model, sample_size, preprocess_data, mode):

    if preprocess_data:
        encode_data(model, data_path, lower=True, reddit=True)
        data_path = data_path.replace('.csv', '_encoded.csv') 

    if mode in ['create', 'project']:
        if not os.path.isfile(representations_path):
            if not os.path.isfile(contexts_path):
                type = 'wordpairs' if mode == 'create' else 'keywords'
                encoded2keyword = encode_queries(model, lu_path, type)
                sample_contexts(data_path, contexts_path, encoded2keyword, sample_size)
            extract_representations(model, contexts_path, representations_path, 
                                    layer_selection='all', layer_aggregation='mean')

    if mode == 'create':
        create_dimension(representations_path, dimension_path, lu_path, method='kozlowski', sample=None)
    elif mode == 'project':
        assert os.path.isfile(dimension_path)
        dimension_projection(representations_path, dimension_path, projections_path)


if __name__ == '__main__':

    lu_path = '' # wordpairs for creation or testwords for projection
    data_path = ''  # data for context sampling 
    
    preprocess_data = False
    mode = 'project' # 'create' or 'project'

    model = 'distilbert-base-uncased' 
    sample_size = None # number of contexts to include per word
    
    contexts_path = ''
    representations_path = ''
    
    dimension_path = ''
    projections_path = ''
        
    main(data_path, lu_path, contexts_path, dimension_path, representations_path, 
            projections_path, model, sample_size, preprocess_data, mode)