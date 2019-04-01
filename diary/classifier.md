### Classifier Metrics 
####Classifier Dropdown
Better results given with the SVC C=600
```
              precision    recall  f1-score   support

         0.0       0.79      0.91      0.85      2358
         1.0       0.95      0.88      0.91      4520

   micro avg       0.89      0.89      0.89      6878
   macro avg       0.87      0.89      0.88      6878
weighted avg       0.90      0.89      0.89      6878
```

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
SVC ,  C= 40
****Results****
Accuracy: 82.6839%
Log Loss: 0.3549574966814722
==============================
SVC ,  C= 70
****Results****
Accuracy: 84.3123%
Log Loss: 0.33818770125357156
==============================
SVC ,  C= 90
****Results****
Accuracy: 84.9229%
Log Loss: 0.33291331690653253
==============================
SVC ,  C= 120
****Results****
Accuracy: 85.6499%
Log Loss: 0.3283367858640148
==============================
SVC ,  C= 140
****Results****
Accuracy: 86.0279%
Log Loss: 0.3254150902970894
==============================
SVC ,  C= 160
****Results****
Accuracy: 86.3332%
Log Loss: 0.3225213417492179
==============================
SVC ,  C= 180
****Results****
Accuracy: 86.5513%
Log Loss: 0.32042467142373987
==============================
SVC ,  C= 200
****Results****
Accuracy: 86.7549%
Log Loss: 0.3187305933611811
==============================
SVC ,  C= 220
****Results****
Accuracy: 86.9003%
Log Loss: 0.31727435043484925
==============================
SVC ,  C= 240
****Results****
Accuracy: 87.1038%
Log Loss: 0.3158971170111406
==============================
SVC ,  C= 260
****Results****
Accuracy: 87.2201%
Log Loss: 0.31467225882758204
==============================
SVC ,  C= 280
****Results****
Accuracy: 87.3510%
Log Loss: 0.3134513013677067
==============================
SVC ,  C= 300
****Results****
Accuracy: 87.4964%
Log Loss: 0.31315646279201215
==============================
SVC ,  C= 320
****Results****
Accuracy: 87.5254%
Log Loss: 0.3122163580303729
==============================
SVC ,  C= 340
****Results****
Accuracy: 87.5545%
Log Loss: 0.31205864216665985
==============================
SVC ,  C= 360
****Results****
Accuracy: 87.7290%
Log Loss: 0.31121470822981834
==============================
SVC ,  C= 380
****Results****
Accuracy: 87.8308%
Log Loss: 0.31040603332549765
==============================
SVC ,  C= 400
****Results****
Accuracy: 87.8598%
Log Loss: 0.3096418506892288
==============================
SVC ,  C= 450
****Results****
Accuracy: 87.9762%
Log Loss: 0.30785329628070235
==============================
SVC ,  C= 500
****Results****
Accuracy: 88.1506%
Log Loss: 0.3070766063438373
==============================
SVC ,  C= 550
****Results****
Accuracy: 88.2524%
Log Loss: 0.30501844522809785
==============================
SVC ,  C= 600
****Results****
Accuracy: 88.8049%
Log Loss: 0.30509675165393546
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



