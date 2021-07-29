# Language Detection

## How To Use

### Train
You can run the following code in your terminal:

`python3 main.py --action train`

This command has several steps.
* Load the data
* Building Vocab from train set
* Find the best classifier from 3 options
    * LogisticRegression
    * SVC
    * MultinomialNB
        * Best model = the highest mean score of 5 runs.
* Train the best classifier
* Evaluate the best classifier on test set 
* Print F1, Accuracy, Recall, Precision
* Saves the model

### Evaluate on a file
You can run the following code in your terminal:

`python3 main.py --action eval --file_to_predict <Path To File>`

The code will transform the text to array with the old vocab.

Predict the language based on the last best model.



