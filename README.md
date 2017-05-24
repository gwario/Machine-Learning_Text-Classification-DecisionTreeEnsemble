# TextClassification

## Configuration
The pipeline configuration for the multi-class and binary data-set can by applied separately in *config.py*.
The pipeline can be configure with the variables *binary_pipeline*/*multiclass_pipeline*, the parameters for the pipeline with the variables *binary_pipeline_parameters*/*multiclass_pipeline_parameters* and the parameter grid (for *--grid*) with the variables *binary_pipeline_parameters_grid*/*multiclass_pipeline_parameters_grid*

## Usage
    usage: main.py [-h] [--train FILE] [--grid] [--model FILE] [--predict FILE]

    optional arguments:
      -h, --help      show this help message and exit
      --train FILE    The training data to be used to create a model. The created
                      model <timestamp>.model is savede to disk.
      --grid          Whether or not to use grid search to get the optimal hyper-
                      parameter configuration. See http://scikit-
                      learn.org/stable/modules/grid_search.html#grid-search
      --model FILE    The model to be used for classification.
      --predict FILE  Data to be classified.

### Display help/usage:
    $ python ./main.py -h

### Training with hyper parameter search and classification:
    $ python ./main.py --train train.csv --grid  --predict testLimited.csv
    
### Training and classification:
    $ python ./main.py --train train.csv --predict testLimited.csv
    
### Training with hyper parameter search:
    $ python ./main.py --train train.csv --grid
    
### Training:
    $ python ./main.py --train train.csv

### Classification with a predefined model:
    $ python ./main.py --model model_<timestamp>.pkl --predict testLimited.csv