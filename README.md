# TextClassification

## Configuration
The pipeline configuration for the multi-class and binary data-set can by applied separately in *config.py*.
The pipeline can be configure with the variables *binary_pipeline*/*multiclass_pipeline*, the parameters for the pipeline with the variables *binary_pipeline_parameters*/*multiclass_pipeline_parameters* and the parameter grid (for *--grid*) with the variables *binary_pipeline_parameters_grid*/*multiclass_pipeline_parameters_grid*

## Usage
    usage: main.py [-h] [--train FILE] [--hp [METHOD]] [--hp_metric [METRIC]]
                   [--score] [--test_size FRACTION] [--model FILE]
                   [--predict FILE]
    
    Classifies, train and saves the model.
    
    Modes of operation:
    
    Score, train and predict:   (--score --train --predict)
        Evaluates the classifier, trains it (model saved to disk) and predicts (prediction saved to disk).
        
    Score and train:            (--score --train)
        Evaluates the classifier and trains it (model saved to disk).
        
    Train and predict:          (--train --predict)
        Trains the classifier (model saved to disk) and predicts (prediction saved to disk).
        
    Load model and predict:     (--model --predict)
        Loads the model and predicts (prediction saved to disk).
        
    All modes of operation support model selection with --hp.
    
    optional arguments:
      -h, --help            show this help message and exit
      --train FILE          The training data to be used to create a model. The
                            created model <timestamp>.model is saved to disk.
      --hp [METHOD]         The method to get the hyper-parameters. One of
                            'config' (use the pre-defined configuration in
                            config.py) or 'grid' (GridSearchCV). (default:
                            'config'
      --hp_metric [METRIC]  The metric to use for the hyper-parameter
                            optimization. Used with 'grid'. (default: 'f1_macro'
      --score               Whether or not to evaluate the estimator performance.
      --test_size FRACTION  Size of the test/train split, as a fraction of the
                            total data.
      --model FILE          The model to be used for classification.
      --predict FILE        Data to be classified.

### Display help/usage:
    $ python ./main.py -h

### Training with hyper parameter search and classification:
    $ python ./main.py --train train.csv --hp grid --predict testLimited.csv
    
### Training and classification:
    $ python ./main.py --train train.csv --predict testLimited.csv
    
### Training with hyper parameter search:
    $ python ./main.py --train train.csv --hp grid
    
### Score and Training:
    $ python ./main.py --train train.csv --score

### Classification with a predefined model:
    $ python ./main.py --model model_<timestamp>.pkl --predict testLimited.csv

