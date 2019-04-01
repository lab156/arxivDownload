### Classifier Metrics 
####Classifier Dropdown
Classifier Dropdown using code from https://www.kaggle.com/jeffd23/10-classifier-showdown-in-scikit-learn
Best results are given by Naive Bayes apparently
```
==============================
MultinomialNB
****Results****
Accuracy: 86.1733%
Log Loss: 1.8957941996159562
==============================
KNeighborsClassifier (with 2 neighbors)
****Results****
Accuracy: 68.4211%
Log Loss: 7.563395488698716
==============================
NuSVC
****Results****
Accuracy: 84.0215%
Log Loss: 0.34342025343628446
==============================
DecisionTreeClassifier
****Results****
Accuracy: 81.3609%
Log Loss: 6.437730639470123
==============================
RandomForestClassifier
****Results****
Accuracy: 83.6580%
Log Loss: 0.6044826387423514
==============================
AdaBoostClassifier
****Results****
Accuracy: 84.4868%
Log Loss: 0.6717126298133219
==============================
GradientBoostingClassifier
****Results****
Accuracy: 85.8098%
Log Loss: 0.3531810109520398
```

Support Vector Classiffier, really slow to train
```
SVC ,  C= 20
****Results****
Accuracy: 80.4304%
Log Loss: 0.381711106061162
==============================
SVC ,  C= 30
****Results****
Accuracy: 81.8552%
Log Loss: 0.3652657882095485
==============================
SVC ,  C= 40
****Results****
Accuracy: 82.6839%
Log Loss: 0.3549574966814722
==============================
SVC ,  C= 50
****Results****
Accuracy: 83.2364%
Log Loss: 0.347609073635744
==============================
SVC ,  C= 70
****Results****
Accuracy: 84.3123%
Log Loss: 0.33818770125357156
==============================
```

This is more or less third attempt just an increase of the data available with html data
* Multinomial Naive Bayes
* math and citation replacement
* 16,532 defs and 11,000 nondefs
```
          precision    recall  f1-score   support

         0.0       0.73      0.91      0.81      2217
         1.0       0.95      0.84      0.89      4661

   micro avg       0.86      0.86      0.86      6878
   macro avg       0.84      0.87      0.85      6878
weighted avg       0.88      0.86      0.87      6878
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


### Pipeline

The *sampling.py* function creates the negative and positive laber at the same time. 
This function has the advantage that it checks if random paragraph are _not_ labeled as definitions.
Usage:
```python
 python sampling.py ../sample18/defs.txt ../sample18/nondefs.txt
```
The time interval and the sampling parameters are currently hardcoded. 

The _positive labels_ are generated with the parsing_xml.py script. An example:
```python
     python parsing_xml.py ~/media_home/math.AG/2015/*/*.xml ../new_real_defs.txt -l ../errors_new_real_defs.txt 
```
Where the l flag stands for logging and is necessary to finish processin all the files


The _negative labels_ can be obtained by random samples with the script:
```python
      python random_sampling.py ~/media_home/math.AG/2015/*/*.xml -o ../new_nondefs.txt
```



