import os, argparse
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

    parser = argparse.ArgumentParser()

    parser.add_argument("--lu_path",
                        help="""Path to textfile with lexical pairs (one pair of LU's divided by ';' per line)
                        or path to textfile with test terms (one LU per line)""",
                        type=str)
    parser.add_argument("--data_path",
                        help="""Path to csv-file with data for context sampling, 
                        with the column 'text' containing one context per row""",
                        type=str)
    parser.add_argument("--preprocess_data",
                        help="""Only set to False if data has already been pre-processed before""",
                        default=True,
                        type=bool)
    parser.add_argument("--mode",
                        help="""Set to 'create' to create a dimension based on a set of word pairs.
                        Set to 'project' to project a set of test terms on a created dimension""",
                        choices=['create', 'project'],
                        default='create',
                        type=str)
    parser.add_argument("--model",
                        help="""Directory/name of model available via Hugging Face's transformers library""",
                        default='distilbert-base-uncased',
                        type=str)
    parser.add_argument("--sample_size",
                        help="""Number of contexts to include per LU, set to None if all should be included""",
                        default=None,
                        type=int)
    parser.add_argument("--contexts_path",
                        help="""Path to csv-file for writing the contexts per target LU
                        or existing path to contexts csv-file""",
                        type=str)
    parser.add_argument("--representations_path",
                        help="""Path to file (no extension) for pickle dump LU representations
                        or existing file to pickle load LU representations from""",
                        type=str)
    parser.add_argument("--dimension_path",
                        help="""Path to file (no extension) for pickle dump the dimension to be created
                        or existing file to pickle load dimension to project on""",
                        type=str)
    parser.add_argument("--projections_path",
                        help="""Path to csv-file to write projections values to (if mode == 'project')""",
                        type=str)
    
    args = parser.parse_args()

    main(args.data_path, args.lu_path, args.contexts_path, args.dimension_path, args.representations_path, 
            args.projections_path, args.model, args.sample_size, args.preprocess_data, args.mode)
    