# lstm_double_layer_model trained on math14 and math15 for 5 epochs:
    Optimal probabilty threshold is 0.55 for maximum F1 score 0.929739631034934
                  precision    recall  f1-score   support

             0.0       0.90      0.95      0.93      7143
             1.0       0.96      0.91      0.93      7857

        accuracy                           0.93     15000
       macro avg       0.93      0.93      0.93     15000
    weighted avg       0.93      0.93      0.93     15000

    AUC ROC: 0.9791691303253174
    
# conv_model trained on math14 and math15 for 20 epochs:
 'testing_size': 15000,
 'embed_dim': 200,
 'tot_words': 65827,
 'conv_filters': 128,
 'kernel_size': 5}
    Optimal probabilty threshold is 0.47 for maximum F1 score 0.9179523594345104
                  precision    recall  f1-score   support

             0.0       0.90      0.93      0.91      7156
             1.0       0.93      0.90      0.92      7844

        accuracy                           0.92     15000
       macro avg       0.92      0.92      0.92     15000
    weighted avg       0.92      0.92      0.92     15000

    AUC ROC: 0.9701041579246521