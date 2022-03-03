# IC_NLP_CW
This is the repository for NLP coursework.

## Description of the files
- `trainer.py`: contains our Trainer class
- `loss.py`: contains custom loss functions that we used
- `model.py`: contains the Model class we used.
- `synonyms.py`: contains the functions for data augmentation
- `FGM.py`: Adverserial attack training techniques
- `utils.py`: contains useful functions for data analsis and manipulation
- `main.py`: the main script that runs the experiment
- `models`: directory contains saved models
- `nlp_data`: directory contains data
- `result`: directory contains results of the test set

## Required packages
`transformers          4.16.2`  
`spacy                 3.2.0`  
`torch                 1.10.0`  
`numpy                 1.19.5`  
`pandas                1.1.5`  
`nltk                  3.4.5`  
`scikit-learn          0.24.2`  
`tqdm                  4.62.3`  
`xgboost               1.4.2`  

## How to run the project
* To replicate the experiment, please just place all the data (.csv or .tsv) into the directory `nlp_data` then run `python3 main.py`.

## Important notes
* Please make sure you have all required libraries installed

* Please modify `config` varaible in the `main.py` to test different hyperparameters/tricks. 

* `MyBertModel_2` is the model that includes additional *keyword* information. `MyBertModel` only takes paragraph as inputs Please switch between for different experiments

* The model is saved in `models` directory. The test result will be saved as `task1.txt`

## Reference
Loper, E., & Bird, S. (2002). Nltk: The natural language toolkit. arXiv preprint cs/0205028.

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2020, October). Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations (pp. 38-45).
