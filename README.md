# IC_NLP_CW
This is the repository for NLP coursework.

## Description of the files
- trainer.py: contains our Trainer class
- loss.py: contains custom loss functions that we used
- model.py: contains the Model class we used.
- synonyms.py: contains the functions for data augmentation
- FGM.py: Adverserial attack training techniques
- utils.py: contains useful functions for data analsis and manipulation
- main.py: the main script that runs the experiment

## How to run the project
* To replicate the experiment, please just place the all the data (.csv or .tsv) into the directory `nlp_data` then `python3 main.py`.

## Important notes
* Please make sure you have all required libraries installed

* Please modify `config` varaible in the `main.py` to test different hyperparameters/tricks. 

* `MyBertModel_2` is the model that includes additional *keyword* information. `MyBertModel` only takes paragraph as inputs Please switch between for different experiments

* The model is saved in `models` directory. The test result will be saved as `task1.txt`
