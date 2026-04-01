# results_compare_A1_A4

| experiment | description | best_model | val_mae | val_rmse | test_mae | test_rmse | seq_len | pred_len | feature_count |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | patch internal only | TCN | 4.064265e-02 | 6.922130e-02 | 1.534805e-01 | 5.002355e-01 | 24 | 12 | 10 |
| A2 | A1 + weather | LSTM | 4.303327e-02 | 7.205674e-02 | 1.025435e-01 | 4.548892e-01 | 24 | 12 | 14 |
| A3 | A2 + blast V2 | TCN | 4.573414e-02 | 8.062277e-02 | 1.457417e-01 | 4.898550e-01 | 24 | 12 | 21 |
| A4 | A2 + blast V3 real-coordinate | LSTM | 4.410412e-02 | 8.058664e-02 | 1.006999e-01 | 4.549706e-01 | 24 | 12 | 18 |
