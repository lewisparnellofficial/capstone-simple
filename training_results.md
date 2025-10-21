# No SMOTE, no Chi2

```bash
PS C:\Users\lewis\Documents\WGU\capstone-simple> uv run ids-train --no-smote --no-chi2

Training Configuration:
  Dataset: data/raw/dataset.csv
  Output directory: models
  GPU acceleration: Enabled
  Chi-Squared feature selection: Disabled
  SMOTE: Disabled

==================================================
Training XGBoost Model
==================================================
Using GPU acceleration: Yes

Model parameters:
  colsample_bytree: 0.8
  device: cuda
  early_stopping_rounds: 50
  gamma: 0.1
  learning_rate: 0.1
  max_depth: 5
  min_child_weight: 1
  n_estimators: 100
  num_class: 15
  objective: multi:softprob
  random_state: 42
  reg_alpha: 0.5
  reg_lambda: 2
  subsample: 0.8
  tree_method: hist

[0]     validation_0-mlogloss:1.72321
[1]     validation_0-mlogloss:1.46334
[2]     validation_0-mlogloss:1.26854
[3]     validation_0-mlogloss:1.11421
[4]     validation_0-mlogloss:0.98569
[5]     validation_0-mlogloss:0.87691
[6]     validation_0-mlogloss:0.78414
[7]     validation_0-mlogloss:0.70266
[8]     validation_0-mlogloss:0.63240
[9]     validation_0-mlogloss:0.56962
[10]    validation_0-mlogloss:0.51420
[11]    validation_0-mlogloss:0.46544
[12]    validation_0-mlogloss:0.42143
[13]    validation_0-mlogloss:0.38217
[14]    validation_0-mlogloss:0.34698
[15]    validation_0-mlogloss:0.31471
[16]    validation_0-mlogloss:0.28599
[17]    validation_0-mlogloss:0.26016
[18]    validation_0-mlogloss:0.23663
[19]    validation_0-mlogloss:0.21561
[20]    validation_0-mlogloss:0.19660
[21]    validation_0-mlogloss:0.17917
[22]    validation_0-mlogloss:0.16376
[23]    validation_0-mlogloss:0.14956
[24]    validation_0-mlogloss:0.13663
[25]    validation_0-mlogloss:0.12492
[26]    validation_0-mlogloss:0.11457
[27]    validation_0-mlogloss:0.10507
[28]    validation_0-mlogloss:0.09635
[29]    validation_0-mlogloss:0.08839
[30]    validation_0-mlogloss:0.08126
[31]    validation_0-mlogloss:0.07475
[32]    validation_0-mlogloss:0.06883
[33]    validation_0-mlogloss:0.06347
[34]    validation_0-mlogloss:0.05841
[35]    validation_0-mlogloss:0.05395
[36]    validation_0-mlogloss:0.04978
[37]    validation_0-mlogloss:0.04603
[38]    validation_0-mlogloss:0.04266
[39]    validation_0-mlogloss:0.03944
[40]    validation_0-mlogloss:0.03653
[41]    validation_0-mlogloss:0.03397
[42]    validation_0-mlogloss:0.03158
[43]    validation_0-mlogloss:0.02935
[44]    validation_0-mlogloss:0.02741
[45]    validation_0-mlogloss:0.02561
[46]    validation_0-mlogloss:0.02393
[47]    validation_0-mlogloss:0.02241
[48]    validation_0-mlogloss:0.02100
[49]    validation_0-mlogloss:0.01974
[50]    validation_0-mlogloss:0.01860
[51]    validation_0-mlogloss:0.01756
[52]    validation_0-mlogloss:0.01658
[53]    validation_0-mlogloss:0.01572
[54]    validation_0-mlogloss:0.01489
[55]    validation_0-mlogloss:0.01413
[56]    validation_0-mlogloss:0.01341
[57]    validation_0-mlogloss:0.01276
[58]    validation_0-mlogloss:0.01214
[59]    validation_0-mlogloss:0.01160
[60]    validation_0-mlogloss:0.01112
[61]    validation_0-mlogloss:0.01067
[62]    validation_0-mlogloss:0.01019
[63]    validation_0-mlogloss:0.00974
[64]    validation_0-mlogloss:0.00940
[65]    validation_0-mlogloss:0.00908
[66]    validation_0-mlogloss:0.00875
[67]    validation_0-mlogloss:0.00847
[68]    validation_0-mlogloss:0.00819
[69]    validation_0-mlogloss:0.00790
[70]    validation_0-mlogloss:0.00765
[71]    validation_0-mlogloss:0.00741
[72]    validation_0-mlogloss:0.00720
[73]    validation_0-mlogloss:0.00703
[74]    validation_0-mlogloss:0.00684
[75]    validation_0-mlogloss:0.00667
[76]    validation_0-mlogloss:0.00652
[77]    validation_0-mlogloss:0.00637
[78]    validation_0-mlogloss:0.00623
[79]    validation_0-mlogloss:0.00608
[80]    validation_0-mlogloss:0.00596
[81]    validation_0-mlogloss:0.00585
[82]    validation_0-mlogloss:0.00576
[83]    validation_0-mlogloss:0.00566
[84]    validation_0-mlogloss:0.00557
[85]    validation_0-mlogloss:0.00551
[86]    validation_0-mlogloss:0.00542
[87]    validation_0-mlogloss:0.00534
[88]    validation_0-mlogloss:0.00528
[89]    validation_0-mlogloss:0.00521
[90]    validation_0-mlogloss:0.00515
[91]    validation_0-mlogloss:0.00509
[92]    validation_0-mlogloss:0.00503
[93]    validation_0-mlogloss:0.00498
[94]    validation_0-mlogloss:0.00492
[95]    validation_0-mlogloss:0.00485
[96]    validation_0-mlogloss:0.00481
[97]    validation_0-mlogloss:0.00477
[98]    validation_0-mlogloss:0.00472
[99]    validation_0-mlogloss:0.00468

==================================================
Validation Set Performance
==================================================
Validation Accuracy: 0.9988

                            precision    recall  f1-score   support

                    BENIGN       1.00      1.00      1.00    227005
                       Bot       0.96      0.69      0.80       189
                      DDoS       1.00      1.00      1.00     12725
             DoS GoldenEye       0.99      1.00      0.99      1032
                  DoS Hulk       1.00      1.00      1.00     23092
          DoS Slowhttptest       0.99      0.98      0.98       573
             DoS slowloris       0.99      1.00      1.00       560
               FTP-Patator       1.00      1.00      1.00       804
              Infiltration       1.00      1.00      1.00         4
                  PortScan       0.99      1.00      1.00     15961
               SSH-Patator       1.00      1.00      1.00       619
  Web Attack � Brute Force       0.73      0.97      0.83       157
Web Attack � Sql Injection       0.00      0.00      0.00         1
          Web Attack � XSS       0.67      0.06      0.11        66

                  accuracy                           1.00    282788
                 macro avg       0.88      0.83      0.84    282788
              weighted avg       1.00      1.00      1.00    282788


==================================================
Test Set Performance
==================================================
Test Accuracy: 0.9988

                            precision    recall  f1-score   support

                    BENIGN       1.00      1.00      1.00    227084
                       Bot       0.97      0.62      0.76       197
                      DDoS       1.00      1.00      1.00     12863
             DoS GoldenEye       1.00      0.99      0.99      1023
                  DoS Hulk       1.00      1.00      1.00     23167
          DoS Slowhttptest       0.98      0.98      0.98       530
             DoS slowloris       0.99      1.00      1.00       586
               FTP-Patator       1.00      1.00      1.00       766
                Heartbleed       1.00      1.00      1.00         3
              Infiltration       1.00      1.00      1.00         1
                  PortScan       0.99      1.00      1.00     15751
               SSH-Patator       1.00      1.00      1.00       623
  Web Attack � Brute Force       0.71      0.96      0.82       134
Web Attack � Sql Injection       0.00      0.00      0.00         4
          Web Attack � XSS       0.33      0.04      0.06        56

                  accuracy                           1.00    282788
                 macro avg       0.87      0.84      0.84    282788
              weighted avg       1.00      1.00      1.00    282788

Model saved to models\model.json
Label mapping saved to models\label_mapping.json
Best hyperparameters saved to models\best_hyperparameters.json
```

# SMOTE, no Chi2
```bash
PS C:\Users\lewis\Documents\WGU\capstone-simple> uv run ids-train --no-chi2

Training Configuration:
  Dataset: data/raw/dataset.csv
  Output directory: models
  GPU acceleration: Enabled
  Chi-Squared feature selection: Disabled
  SMOTE: Enabled
  SMOTE neighbors: 5


==================================================
Applying SMOTE to balance training data...
==================================================
Original training size: 2,262,300 samples
Class distribution before SMOTE:
  BENIGN (class 0): 1,817,231 samples (80.33%)
  Bot (class 1): 1,570 samples (0.07%)
  DDoS (class 2): 102,437 samples (4.53%)
  DoS GoldenEye (class 3): 8,238 samples (0.36%)
  DoS Hulk (class 4): 183,865 samples (8.13%)
  DoS Slowhttptest (class 5): 4,396 samples (0.19%)
  DoS slowloris (class 6): 4,650 samples (0.21%)
  FTP-Patator (class 7): 6,365 samples (0.28%)
  Heartbleed (class 8): 8 samples (0.00%)
  Infiltration (class 9): 31 samples (0.00%)
  PortScan (class 10): 127,092 samples (5.62%)
  SSH-Patator (class 11): 4,655 samples (0.21%)
  Web Attack � Brute Force (class 12): 1,216 samples (0.05%)
  Web Attack � Sql Injection (class 13): 16 samples (0.00%)
  Web Attack � XSS (class 14): 530 samples (0.02%)

After SMOTE training size: 27,258,465 samples
Class distribution after SMOTE:
  BENIGN (class 0): 1,817,231 samples (6.67%)
  Bot (class 1): 1,817,231 samples (6.67%)
  DDoS (class 2): 1,817,231 samples (6.67%)
  DoS GoldenEye (class 3): 1,817,231 samples (6.67%)
  DoS Hulk (class 4): 1,817,231 samples (6.67%)
  DoS Slowhttptest (class 5): 1,817,231 samples (6.67%)
  DoS slowloris (class 6): 1,817,231 samples (6.67%)
  FTP-Patator (class 7): 1,817,231 samples (6.67%)
  Heartbleed (class 8): 1,817,231 samples (6.67%)
  Infiltration (class 9): 1,817,231 samples (6.67%)
  PortScan (class 10): 1,817,231 samples (6.67%)
  SSH-Patator (class 11): 1,817,231 samples (6.67%)
  Web Attack � Brute Force (class 12): 1,817,231 samples (6.67%)
  Web Attack � Sql Injection (class 13): 1,817,231 samples (6.67%)
  Web Attack � XSS (class 14): 1,817,231 samples (6.67%)
==================================================

==================================================
Training XGBoost Model
==================================================
Using GPU acceleration: Yes

Model parameters:
  colsample_bytree: 0.8
  device: cuda
  early_stopping_rounds: 50
  gamma: 0.1
  learning_rate: 0.1
  max_depth: 5
  min_child_weight: 1
  n_estimators: 100
  num_class: 15
  objective: multi:softprob
  random_state: 42
  reg_alpha: 0.5
  reg_lambda: 2
  subsample: 0.8
  tree_method: hist

[0]     validation_0-mlogloss:2.10089
[1]     validation_0-mlogloss:1.82905
[2]     validation_0-mlogloss:1.64097
[3]     validation_0-mlogloss:1.48307
[4]     validation_0-mlogloss:1.31416
[5]     validation_0-mlogloss:1.18542
[6]     validation_0-mlogloss:1.06638
[7]     validation_0-mlogloss:0.96639
[8]     validation_0-mlogloss:0.88258
[9]     validation_0-mlogloss:0.81189
[10]    validation_0-mlogloss:0.75388
[11]    validation_0-mlogloss:0.69482
[12]    validation_0-mlogloss:0.63947
[13]    validation_0-mlogloss:0.59312
[14]    validation_0-mlogloss:0.55052
[15]    validation_0-mlogloss:0.50946
[16]    validation_0-mlogloss:0.47255
[17]    validation_0-mlogloss:0.43900
[18]    validation_0-mlogloss:0.40541
[19]    validation_0-mlogloss:0.37811
[20]    validation_0-mlogloss:0.35333
[21]    validation_0-mlogloss:0.33099
[22]    validation_0-mlogloss:0.31015
[23]    validation_0-mlogloss:0.29165
[24]    validation_0-mlogloss:0.27453
[25]    validation_0-mlogloss:0.25842
[26]    validation_0-mlogloss:0.24332
[27]    validation_0-mlogloss:0.22942
[28]    validation_0-mlogloss:0.21575
[29]    validation_0-mlogloss:0.20298
[30]    validation_0-mlogloss:0.19231
[31]    validation_0-mlogloss:0.18184
[32]    validation_0-mlogloss:0.17072
[33]    validation_0-mlogloss:0.16089
[34]    validation_0-mlogloss:0.15065
[35]    validation_0-mlogloss:0.14263
[36]    validation_0-mlogloss:0.13402
[37]    validation_0-mlogloss:0.12678
[38]    validation_0-mlogloss:0.11934
[39]    validation_0-mlogloss:0.11298
[40]    validation_0-mlogloss:0.10649
[41]    validation_0-mlogloss:0.10052
[42]    validation_0-mlogloss:0.09507
[43]    validation_0-mlogloss:0.09051
[44]    validation_0-mlogloss:0.08594
[45]    validation_0-mlogloss:0.08159
[46]    validation_0-mlogloss:0.07791
[47]    validation_0-mlogloss:0.07416
[48]    validation_0-mlogloss:0.07067
[49]    validation_0-mlogloss:0.06702
[50]    validation_0-mlogloss:0.06381
[51]    validation_0-mlogloss:0.06084
[52]    validation_0-mlogloss:0.05813
[53]    validation_0-mlogloss:0.05567
[54]    validation_0-mlogloss:0.05345
[55]    validation_0-mlogloss:0.05122
[56]    validation_0-mlogloss:0.04913
[57]    validation_0-mlogloss:0.04712
[58]    validation_0-mlogloss:0.04510
[59]    validation_0-mlogloss:0.04339
[60]    validation_0-mlogloss:0.04184
[61]    validation_0-mlogloss:0.04034
[62]    validation_0-mlogloss:0.03871
[63]    validation_0-mlogloss:0.03730
[64]    validation_0-mlogloss:0.03612
[65]    validation_0-mlogloss:0.03504
[66]    validation_0-mlogloss:0.03403
[67]    validation_0-mlogloss:0.03291
[68]    validation_0-mlogloss:0.03175
[69]    validation_0-mlogloss:0.03088
[70]    validation_0-mlogloss:0.02992
[71]    validation_0-mlogloss:0.02912
[72]    validation_0-mlogloss:0.02841
[73]    validation_0-mlogloss:0.02755
[74]    validation_0-mlogloss:0.02669
[75]    validation_0-mlogloss:0.02603
[76]    validation_0-mlogloss:0.02539
[77]    validation_0-mlogloss:0.02472
[78]    validation_0-mlogloss:0.02417
[79]    validation_0-mlogloss:0.02362
[80]    validation_0-mlogloss:0.02315
[81]    validation_0-mlogloss:0.02258
[82]    validation_0-mlogloss:0.02213
[83]    validation_0-mlogloss:0.02170
[84]    validation_0-mlogloss:0.02124
[85]    validation_0-mlogloss:0.02083
[86]    validation_0-mlogloss:0.02035
[87]    validation_0-mlogloss:0.01993
[88]    validation_0-mlogloss:0.01964
[89]    validation_0-mlogloss:0.01926
[90]    validation_0-mlogloss:0.01897
[91]    validation_0-mlogloss:0.01870
[92]    validation_0-mlogloss:0.01841
[93]    validation_0-mlogloss:0.01820
[94]    validation_0-mlogloss:0.01788
[95]    validation_0-mlogloss:0.01760
[96]    validation_0-mlogloss:0.01737
[97]    validation_0-mlogloss:0.01706
[98]    validation_0-mlogloss:0.01682
[99]    validation_0-mlogloss:0.01657

==================================================
Validation Set Performance
==================================================
Validation Accuracy: 0.9954

                            precision    recall  f1-score   support

                    BENIGN       1.00      0.99      1.00    227005
                       Bot       0.22      0.99      0.36       189
                      DDoS       1.00      1.00      1.00     12725
             DoS GoldenEye       0.97      1.00      0.98      1032
                  DoS Hulk       0.99      1.00      0.99     23092
          DoS Slowhttptest       0.96      0.99      0.97       573
             DoS slowloris       0.98      1.00      0.99       560
               FTP-Patator       1.00      1.00      1.00       804
              Infiltration       0.57      1.00      0.73         4
                  PortScan       0.99      1.00      1.00     15961
               SSH-Patator       1.00      1.00      1.00       619
  Web Attack � Brute Force       0.63      0.62      0.62       157
Web Attack � Sql Injection       0.11      1.00      0.20         1
          Web Attack � XSS       0.28      0.74      0.40        66

                  accuracy                           1.00    282788
                 macro avg       0.76      0.95      0.80    282788
              weighted avg       1.00      1.00      1.00    282788


==================================================
Test Set Performance
==================================================
Test Accuracy: 0.9951

                            precision    recall  f1-score   support

                    BENIGN       1.00      0.99      1.00    227084
                       Bot       0.22      1.00      0.35       197
                      DDoS       1.00      1.00      1.00     12863
             DoS GoldenEye       0.97      1.00      0.98      1023
                  DoS Hulk       0.99      1.00      0.99     23167
          DoS Slowhttptest       0.96      0.99      0.97       530
             DoS slowloris       0.98      1.00      0.99       586
               FTP-Patator       1.00      1.00      1.00       766
                Heartbleed       1.00      1.00      1.00         3
              Infiltration       0.25      1.00      0.40         1
                  PortScan       0.99      1.00      1.00     15751
               SSH-Patator       0.99      1.00      1.00       623
  Web Attack � Brute Force       0.54      0.60      0.57       134
Web Attack � Sql Injection       0.09      0.50      0.15         4
          Web Attack � XSS       0.20      0.59      0.30        56

                  accuracy                           1.00    282788
                 macro avg       0.75      0.91      0.78    282788
              weighted avg       1.00      1.00      1.00    282788

Model saved to models\model.json
Label mapping saved to models\label_mapping.json
Best hyperparameters saved to models\best_hyperparameters.json
```

# No SMOTE, Chi2
```bash
PS C:\Users\lewis\Documents\WGU\capstone-simple> uv run ids-train --no-smote

Training Configuration:
  Dataset: data/raw/dataset.csv
  Output directory: models
  GPU acceleration: Enabled
  Chi-Squared feature selection: Enabled
  Chi-Squared k features: 20
  SMOTE: Disabled


==================================================
Applying Chi-Squared Feature Selection...
==================================================
Original number of features: 78
Selected 20 features based on Chi-Squared scores

Top selected features:
  Idle Min: 491544.32
  Idle Mean: 477565.46
  Idle Max: 470133.80
  Fwd IAT Max: 447520.34
  Flow IAT Max: 438967.83
  Fwd IAT Std: 316462.24
  FIN Flag Count: 277457.43
  PSH Flag Count: 276326.97
  Packet Length Std: 276116.49
  Bwd Packet Length Std: 274451.25
  ... and 10 more features
==================================================

==================================================
Training XGBoost Model
==================================================
Using GPU acceleration: Yes

Model parameters:
  colsample_bytree: 0.8
  device: cuda
  early_stopping_rounds: 50
  gamma: 0.1
  learning_rate: 0.1
  max_depth: 5
  min_child_weight: 1
  n_estimators: 100
  num_class: 15
  objective: multi:softprob
  random_state: 42
  reg_alpha: 0.5
  reg_lambda: 2
  subsample: 0.8
  tree_method: hist

[0]     validation_0-mlogloss:1.74098
[1]     validation_0-mlogloss:1.48755
[2]     validation_0-mlogloss:1.29789
[3]     validation_0-mlogloss:1.14485
[4]     validation_0-mlogloss:1.01869
[5]     validation_0-mlogloss:0.91177
[6]     validation_0-mlogloss:0.81987
[7]     validation_0-mlogloss:0.73996
[8]     validation_0-mlogloss:0.66989
[9]     validation_0-mlogloss:0.60807
[10]    validation_0-mlogloss:0.55280
[11]    validation_0-mlogloss:0.50421
[12]    validation_0-mlogloss:0.46048
[13]    validation_0-mlogloss:0.42142
[14]    validation_0-mlogloss:0.38636
[15]    validation_0-mlogloss:0.35478
[16]    validation_0-mlogloss:0.32634
[17]    validation_0-mlogloss:0.30055
[18]    validation_0-mlogloss:0.27727
[19]    validation_0-mlogloss:0.25641
[20]    validation_0-mlogloss:0.23739
[21]    validation_0-mlogloss:0.22017
[22]    validation_0-mlogloss:0.20476
[23]    validation_0-mlogloss:0.19075
[24]    validation_0-mlogloss:0.17776
[25]    validation_0-mlogloss:0.16616
[26]    validation_0-mlogloss:0.15548
[27]    validation_0-mlogloss:0.14570
[28]    validation_0-mlogloss:0.13691
[29]    validation_0-mlogloss:0.12888
[30]    validation_0-mlogloss:0.12153
[31]    validation_0-mlogloss:0.11490
[32]    validation_0-mlogloss:0.10883
[33]    validation_0-mlogloss:0.10331
[34]    validation_0-mlogloss:0.09831
[35]    validation_0-mlogloss:0.09374
[36]    validation_0-mlogloss:0.08952
[37]    validation_0-mlogloss:0.08570
[38]    validation_0-mlogloss:0.08214
[39]    validation_0-mlogloss:0.07896
[40]    validation_0-mlogloss:0.07611
[41]    validation_0-mlogloss:0.07344
[42]    validation_0-mlogloss:0.07098
[43]    validation_0-mlogloss:0.06869
[44]    validation_0-mlogloss:0.06663
[45]    validation_0-mlogloss:0.06470
[46]    validation_0-mlogloss:0.06289
[47]    validation_0-mlogloss:0.06128
[48]    validation_0-mlogloss:0.05980
[49]    validation_0-mlogloss:0.05847
[50]    validation_0-mlogloss:0.05723
[51]    validation_0-mlogloss:0.05604
[52]    validation_0-mlogloss:0.05497
[53]    validation_0-mlogloss:0.05396
[54]    validation_0-mlogloss:0.05295
[55]    validation_0-mlogloss:0.05213
[56]    validation_0-mlogloss:0.05132
[57]    validation_0-mlogloss:0.05051
[58]    validation_0-mlogloss:0.04983
[59]    validation_0-mlogloss:0.04918
[60]    validation_0-mlogloss:0.04851
[61]    validation_0-mlogloss:0.04790
[62]    validation_0-mlogloss:0.04735
[63]    validation_0-mlogloss:0.04684
[64]    validation_0-mlogloss:0.04641
[65]    validation_0-mlogloss:0.04595
[66]    validation_0-mlogloss:0.04553
[67]    validation_0-mlogloss:0.04508
[68]    validation_0-mlogloss:0.04474
[69]    validation_0-mlogloss:0.04438
[70]    validation_0-mlogloss:0.04406
[71]    validation_0-mlogloss:0.04376
[72]    validation_0-mlogloss:0.04345
[73]    validation_0-mlogloss:0.04318
[74]    validation_0-mlogloss:0.04294
[75]    validation_0-mlogloss:0.04268
[76]    validation_0-mlogloss:0.04248
[77]    validation_0-mlogloss:0.04228
[78]    validation_0-mlogloss:0.04201
[79]    validation_0-mlogloss:0.04172
[80]    validation_0-mlogloss:0.04154
[81]    validation_0-mlogloss:0.04130
[82]    validation_0-mlogloss:0.04106
[83]    validation_0-mlogloss:0.04090
[84]    validation_0-mlogloss:0.04078
[85]    validation_0-mlogloss:0.04058
[86]    validation_0-mlogloss:0.04043
[87]    validation_0-mlogloss:0.04024
[88]    validation_0-mlogloss:0.04012
[89]    validation_0-mlogloss:0.03998
[90]    validation_0-mlogloss:0.03987
[91]    validation_0-mlogloss:0.03971
[92]    validation_0-mlogloss:0.03961
[93]    validation_0-mlogloss:0.03947
[94]    validation_0-mlogloss:0.03936
[95]    validation_0-mlogloss:0.03919
[96]    validation_0-mlogloss:0.03908
[97]    validation_0-mlogloss:0.03893
[98]    validation_0-mlogloss:0.03886
[99]    validation_0-mlogloss:0.03878

==================================================
Validation Set Performance
==================================================
Validation Accuracy: 0.9845

                            precision    recall  f1-score   support

                    BENIGN       0.99      0.99      0.99    227005
                       Bot       0.97      0.37      0.54       189
                      DDoS       0.99      0.99      0.99     12725
             DoS GoldenEye       1.00      0.97      0.98      1032
                  DoS Hulk       0.89      0.97      0.93     23092
          DoS Slowhttptest       0.95      0.98      0.96       573
             DoS slowloris       0.99      0.99      0.99       560
               FTP-Patator       0.99      0.98      0.99       804
              Infiltration       1.00      0.75      0.86         4
                  PortScan       0.99      1.00      1.00     15961
               SSH-Patator       1.00      0.49      0.66       619
  Web Attack � Brute Force       0.95      0.13      0.22       157
Web Attack � Sql Injection       0.00      0.00      0.00         1
          Web Attack � XSS       0.00      0.00      0.00        66

                  accuracy                           0.98    282788
                 macro avg       0.84      0.69      0.72    282788
              weighted avg       0.98      0.98      0.98    282788


==================================================
Test Set Performance
==================================================
Test Accuracy: 0.9844

                            precision    recall  f1-score   support

                    BENIGN       0.99      0.99      0.99    227084
                       Bot       0.96      0.26      0.41       197
                      DDoS       0.99      0.99      0.99     12863
             DoS GoldenEye       1.00      0.96      0.98      1023
                  DoS Hulk       0.89      0.97      0.93     23167
          DoS Slowhttptest       0.93      0.98      0.95       530
             DoS slowloris       0.99      0.99      0.99       586
               FTP-Patator       0.99      0.99      0.99       766
                Heartbleed       1.00      1.00      1.00         3
              Infiltration       1.00      1.00      1.00         1
                  PortScan       0.99      1.00      1.00     15751
               SSH-Patator       1.00      0.48      0.65       623
  Web Attack � Brute Force       1.00      0.10      0.19       134
Web Attack � Sql Injection       0.00      0.00      0.00         4
          Web Attack � XSS       0.00      0.00      0.00        56

                  accuracy                           0.98    282788
                 macro avg       0.85      0.71      0.74    282788
              weighted avg       0.98      0.98      0.98    282788

Model saved to models\model.json
Label mapping saved to models\label_mapping.json
Best hyperparameters saved to models\best_hyperparameters.json
```

# SMOTE, Chi2
```bash
PS C:\Users\lewis\Documents\WGU\capstone-simple> uv run ids-train

Training Configuration:
  Dataset: data/raw/dataset.csv
  Output directory: models
  GPU acceleration: Enabled
  Chi-Squared feature selection: Enabled
  Chi-Squared k features: 20
  SMOTE: Enabled
  SMOTE neighbors: 5


==================================================
Applying Chi-Squared Feature Selection...
==================================================
Original number of features: 78
Selected 20 features based on Chi-Squared scores

Top selected features:
  Idle Min: 491544.32
  Idle Mean: 477565.46
  Idle Max: 470133.80
  Fwd IAT Max: 447520.34
  Flow IAT Max: 438967.83
  Fwd IAT Std: 316462.24
  FIN Flag Count: 277457.43
  PSH Flag Count: 276326.97
  Packet Length Std: 276116.49
  Bwd Packet Length Std: 274451.25
  ... and 10 more features
==================================================


==================================================
Applying SMOTE to balance training data...
==================================================
Original training size: 2,262,300 samples
Class distribution before SMOTE:
  BENIGN (class 0): 1,817,231 samples (80.33%)
  Bot (class 1): 1,570 samples (0.07%)
  DDoS (class 2): 102,437 samples (4.53%)
  DoS GoldenEye (class 3): 8,238 samples (0.36%)
  DoS Hulk (class 4): 183,865 samples (8.13%)
  DoS Slowhttptest (class 5): 4,396 samples (0.19%)
  DoS slowloris (class 6): 4,650 samples (0.21%)
  FTP-Patator (class 7): 6,365 samples (0.28%)
  Heartbleed (class 8): 8 samples (0.00%)
  Infiltration (class 9): 31 samples (0.00%)
  PortScan (class 10): 127,092 samples (5.62%)
  SSH-Patator (class 11): 4,655 samples (0.21%)
  Web Attack � Brute Force (class 12): 1,216 samples (0.05%)
  Web Attack � Sql Injection (class 13): 16 samples (0.00%)
  Web Attack � XSS (class 14): 530 samples (0.02%)

After SMOTE training size: 27,258,465 samples
Class distribution after SMOTE:
  BENIGN (class 0): 1,817,231 samples (6.67%)
  Bot (class 1): 1,817,231 samples (6.67%)
  DDoS (class 2): 1,817,231 samples (6.67%)
  DoS GoldenEye (class 3): 1,817,231 samples (6.67%)
  DoS Hulk (class 4): 1,817,231 samples (6.67%)
  DoS Slowhttptest (class 5): 1,817,231 samples (6.67%)
  DoS slowloris (class 6): 1,817,231 samples (6.67%)
  FTP-Patator (class 7): 1,817,231 samples (6.67%)
  Heartbleed (class 8): 1,817,231 samples (6.67%)
  Infiltration (class 9): 1,817,231 samples (6.67%)
  PortScan (class 10): 1,817,231 samples (6.67%)
  SSH-Patator (class 11): 1,817,231 samples (6.67%)
  Web Attack � Brute Force (class 12): 1,817,231 samples (6.67%)
  Web Attack � Sql Injection (class 13): 1,817,231 samples (6.67%)
  Web Attack � XSS (class 14): 1,817,231 samples (6.67%)
==================================================

==================================================
Training XGBoost Model
==================================================
Using GPU acceleration: Yes

Model parameters:
  colsample_bytree: 0.8
  device: cuda
  early_stopping_rounds: 50
  gamma: 0.1
  learning_rate: 0.1
  max_depth: 5
  min_child_weight: 1
  n_estimators: 100
  num_class: 15
  objective: multi:softprob
  random_state: 42
  reg_alpha: 0.5
  reg_lambda: 2
  subsample: 0.8
  tree_method: hist

[0]     validation_0-mlogloss:2.22307
[1]     validation_0-mlogloss:1.94435
[2]     validation_0-mlogloss:1.73984
[3]     validation_0-mlogloss:1.57718
[4]     validation_0-mlogloss:1.44901
[5]     validation_0-mlogloss:1.33190
[6]     validation_0-mlogloss:1.23519
[7]     validation_0-mlogloss:1.15225
[8]     validation_0-mlogloss:1.08037
[9]     validation_0-mlogloss:1.01529
[10]    validation_0-mlogloss:0.95992
[11]    validation_0-mlogloss:0.91006
[12]    validation_0-mlogloss:0.86589
[13]    validation_0-mlogloss:0.82480
[14]    validation_0-mlogloss:0.78852
[15]    validation_0-mlogloss:0.75519
[16]    validation_0-mlogloss:0.72394
[17]    validation_0-mlogloss:0.69606
[18]    validation_0-mlogloss:0.67040
[19]    validation_0-mlogloss:0.64716
[20]    validation_0-mlogloss:0.62672
[21]    validation_0-mlogloss:0.60815
[22]    validation_0-mlogloss:0.58964
[23]    validation_0-mlogloss:0.57168
[24]    validation_0-mlogloss:0.55619
[25]    validation_0-mlogloss:0.53807
[26]    validation_0-mlogloss:0.52319
[27]    validation_0-mlogloss:0.51029
[28]    validation_0-mlogloss:0.49755
[29]    validation_0-mlogloss:0.48633
[30]    validation_0-mlogloss:0.47616
[31]    validation_0-mlogloss:0.46579
[32]    validation_0-mlogloss:0.45695
[33]    validation_0-mlogloss:0.44840
[34]    validation_0-mlogloss:0.44060
[35]    validation_0-mlogloss:0.43130
[36]    validation_0-mlogloss:0.42476
[37]    validation_0-mlogloss:0.41895
[38]    validation_0-mlogloss:0.41285
[39]    validation_0-mlogloss:0.40726
[40]    validation_0-mlogloss:0.40069
[41]    validation_0-mlogloss:0.39413
[42]    validation_0-mlogloss:0.38883
[43]    validation_0-mlogloss:0.38337
[44]    validation_0-mlogloss:0.37885
[45]    validation_0-mlogloss:0.37365
[46]    validation_0-mlogloss:0.36847
[47]    validation_0-mlogloss:0.36516
[48]    validation_0-mlogloss:0.36147
[49]    validation_0-mlogloss:0.35773
[50]    validation_0-mlogloss:0.35449
[51]    validation_0-mlogloss:0.35059
[52]    validation_0-mlogloss:0.34749
[53]    validation_0-mlogloss:0.34387
[54]    validation_0-mlogloss:0.34143
[55]    validation_0-mlogloss:0.33896
[56]    validation_0-mlogloss:0.33634
[57]    validation_0-mlogloss:0.33374
[58]    validation_0-mlogloss:0.33126
[59]    validation_0-mlogloss:0.32897
[60]    validation_0-mlogloss:0.32667
[61]    validation_0-mlogloss:0.32453
[62]    validation_0-mlogloss:0.32215
[63]    validation_0-mlogloss:0.32022
[64]    validation_0-mlogloss:0.31821
[65]    validation_0-mlogloss:0.31612
[66]    validation_0-mlogloss:0.31427
[67]    validation_0-mlogloss:0.31227
[68]    validation_0-mlogloss:0.31048
[69]    validation_0-mlogloss:0.30858
[70]    validation_0-mlogloss:0.30705
[71]    validation_0-mlogloss:0.30538
[72]    validation_0-mlogloss:0.30387
[73]    validation_0-mlogloss:0.30241
[74]    validation_0-mlogloss:0.30081
[75]    validation_0-mlogloss:0.29976
[76]    validation_0-mlogloss:0.29883
[77]    validation_0-mlogloss:0.29770
[78]    validation_0-mlogloss:0.29639
[79]    validation_0-mlogloss:0.29528
[80]    validation_0-mlogloss:0.29413
[81]    validation_0-mlogloss:0.29284
[82]    validation_0-mlogloss:0.29206
[83]    validation_0-mlogloss:0.29115
[84]    validation_0-mlogloss:0.28985
[85]    validation_0-mlogloss:0.28896
[86]    validation_0-mlogloss:0.28800
[87]    validation_0-mlogloss:0.28726
[88]    validation_0-mlogloss:0.28614
[89]    validation_0-mlogloss:0.28520
[90]    validation_0-mlogloss:0.28446
[91]    validation_0-mlogloss:0.28366
[92]    validation_0-mlogloss:0.28299
[93]    validation_0-mlogloss:0.28180
[94]    validation_0-mlogloss:0.28121
[95]    validation_0-mlogloss:0.28061
[96]    validation_0-mlogloss:0.27994
[97]    validation_0-mlogloss:0.27927
[98]    validation_0-mlogloss:0.27856
[99]    validation_0-mlogloss:0.27806

==================================================
Validation Set Performance
==================================================
Validation Accuracy: 0.9056

                            precision    recall  f1-score   support

                    BENIGN       1.00      0.88      0.94    227005
                       Bot       0.03      0.99      0.05       189
                      DDoS       0.98      1.00      0.99     12725
             DoS GoldenEye       0.67      1.00      0.80      1032
                  DoS Hulk       0.83      0.99      0.90     23092
          DoS Slowhttptest       0.73      0.98      0.84       573
             DoS slowloris       0.88      1.00      0.93       560
               FTP-Patator       0.88      1.00      0.94       804
              Infiltration       0.01      1.00      0.01         4
                  PortScan       0.99      1.00      1.00     15961
               SSH-Patator       0.10      0.83      0.17       619
  Web Attack � Brute Force       0.07      0.50      0.12       157
Web Attack � Sql Injection       0.00      1.00      0.00         1
          Web Attack � XSS       0.03      0.74      0.05        66

                  accuracy                           0.91    282788
                 macro avg       0.51      0.92      0.55    282788
              weighted avg       0.98      0.91      0.94    282788


==================================================
Test Set Performance
==================================================
Test Accuracy: 0.9056

                            precision    recall  f1-score   support

                    BENIGN       1.00      0.88      0.94    227084
                       Bot       0.03      1.00      0.05       197
                      DDoS       0.98      1.00      0.99     12863
             DoS GoldenEye       0.66      0.99      0.79      1023
                  DoS Hulk       0.82      0.99      0.90     23167
          DoS Slowhttptest       0.74      0.99      0.85       530
             DoS slowloris       0.87      1.00      0.93       586
               FTP-Patator       0.88      1.00      0.93       766
                Heartbleed       1.00      1.00      1.00         3
              Infiltration       0.00      1.00      0.00         1
                  PortScan       0.99      1.00      1.00     15751
               SSH-Patator       0.09      0.80      0.17       623
  Web Attack � Brute Force       0.05      0.44      0.09       134
Web Attack � Sql Injection       0.00      0.75      0.00         4
          Web Attack � XSS       0.02      0.71      0.04        56

                  accuracy                           0.91    282788
                 macro avg       0.54      0.90      0.58    282788
              weighted avg       0.98      0.91      0.94    282788

Model saved to models\model.json
Label mapping saved to models\label_mapping.json
Best hyperparameters saved to models\best_hyperparameters.json
```

# Discussion
SMOTE and Chi2 appear to degrade the model performance.

Considering that there is minimal training time difference gained by applying Chi2, then it can be concluded that its not necessary, on this dataset at least.