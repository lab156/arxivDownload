### NER IOB Metrics
This is the first attempt using
* Only wikipedia en data
* Using the ChunkParsr method 
* The features functions does not have Tf-Idf
* Seems to be overfitting because there is a lot of variability between different random samples with the same data.
ChunkParse score:
    IOB Accuracy:  91.1%%
    Precision:     31.5%%
    Recall:        67.6%%
    F-Measure:     43.0%%

The ordinary metric report:
             precision    recall  f1-score   support

    B-DFNDUM       0.35      0.75      0.48      1313
    I-DFNDUM       0.31      0.80      0.44      1022
           O       0.99      0.92      0.95     44020

   micro avg       0.91      0.91      0.91     46355
   macro avg       0.55      0.82      0.62     46355
weighted avg       0.96      0.91      0.93     46355

####Parsing the data:
The latest version is wiki_definitions_improved.txt
