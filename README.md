# SlurDimension

Repository for the paper: Sanne Hoeken, Sina Zarrieß and Özge Alaçam. 2023. Identifying Slurs and Lexical Hate Speech via Light-Weight Dimension Projection in Embedding Space. [Manuscript submitted for publication].
Questions can be directed to sanne.hoeken@uni-bielefeld.de

## Using the main program

Change the working directory of the execution environment to the **/code** directory.
Creating or projection onto a dimension can be run using the following instruction:
  ```
    python3 main.py --lu_path [FILEPATH] --data_path [FILEPATH] --preprocess_data [TRUE or FALSE] --mode ["create" or "project"] --model [MODEL NAME] --sample_size [SAMPLE SIZE] --contexts_path [FILEPATH] --representations_path [FILEPATH] --dimension_path [FILEPATH] --projections_path [FILEPATH]
  ```