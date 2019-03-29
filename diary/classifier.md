### Classifier Metrics 
This is the first attempt
* Multinomial Naive Bayes 
* No math replacement

```
              precision    recall  f1-score   support

         0.0       0.70      0.84      0.76       845
         1.0       0.95      0.89      0.92      2653

   micro avg       0.88      0.88      0.88      3498
   macro avg       0.82      0.86      0.84      3498
weighted avg       0.89      0.88      0.88      3498
```

Second attempt just a cleanup of the data, significant decrease in precision
```
             precision    recall  f1-score   support

         0.0       0.66      0.89      0.76       426
         1.0       0.96      0.87      0.91      1442

   micro avg       0.87      0.87      0.87      1868
   macro avg       0.81      0.88      0.84      1868
weighted avg       0.89      0.87      0.88      1868
```

### Pipeline

The _positive labels_ are generated with the parsing_xml.py script. An example:
```python
     python parsing_xml.py ~/media_home/math.AG/2015/*/*.xml ../new_real_defs.txt -l ../errors_new_real_defs.txt 
```
Where the l flag stands for logging and is necessary to finish processin all the files


The _negative labels_ can be obtained by random samples with the script:
```python
      python random_sampling.py ~/media_home/math.AG/2015/*/*.xml -o ../new_nondefs.txt
```



