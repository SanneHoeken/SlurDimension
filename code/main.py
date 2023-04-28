import os
from preprocess.encode_data import encode_data
from preprocess.encode_queries import encode_queries
from dimension.sample_contexts import sample_contexts
from dimension.extract_representations import extract_representations
from dimension.create_dimension import create_dimension
from dimension.dimension_projection import dimension_projection

def main(data_path, queries_path, 
         contexts_path, dimension_path, 
         representations_path, projections_path,
         type, model, sample_size, layer_selection,
         preprocess_data, preprocess_queries,
         create, project):

    if preprocess_data:
        encode_data(model, data_path, lower=True, reddit=True)
        data_path = data_path.replace('.csv', '_encoded.csv') 
    if preprocess_queries:  
        encode_queries(model, queries_path, type)
        queries_path = queries_path.replace('.txt', '_encoded.txt')

    if create or project:
        if not os.path.isfile(representations_path):
            if not os.path.isfile(contexts_path):
                sample_contexts(data_path, contexts_path, queries_path, type, sample_size)
            extract_representations(model, contexts_path, representations_path, 
                                    layer_selection=layer_selection, layer_aggregation='mean')

    if create:
        create_dimension(representations_path, dimension_path, queries_path, method='kozlowski')
    if project:
        assert os.path.isfile(dimension_path)
        dimension_projection(representations_path, dimension_path, projections_path)

if __name__ == '__main__':

    # DATAFILES
    data_path = '' 
    contexts_path = ''
    queries_path = ''
    representations_path = ''
    dimension_path = ''
    projections_path = ''

    # OTHER PARAMETERS
    type = 'keywords' # or 'wordpairs'
    model = 'distilbert-base-uncased' 
    sample_size = None
    layer_selection = 'all'
    
    # PIPELINE 
    preprocess_data = False
    preprocess_queries = False
    create = False
    project = True
    
    main(data_path, queries_path, 
        contexts_path, dimension_path, 
        representations_path, projections_path,
        type, model, sample_size, layer_selection,
        preprocess_data, preprocess_queries,
        create, project)