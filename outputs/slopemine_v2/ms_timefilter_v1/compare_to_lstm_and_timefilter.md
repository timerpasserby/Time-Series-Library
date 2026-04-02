# compare_to_lstm_and_timefilter

| experiment_id | branch_name | model_name | val_mae | test_mae | blast_test_mae |
| --- | --- | --- | ---: | ---: | ---: |
| C1 | blast_opt | MS-TimeFilter | 0.058301 | 0.111800 | 0.042384 |
| C1 | main | MS-TimeFilter | 0.055784 | 0.120769 | 0.044829 |
| C2 | blast_opt | MS-TimeFilter | 0.055841 | 0.120503 | 0.047920 |
| C2 | main | MS-TimeFilter | 0.050130 | 0.108187 | 0.041348 |
| C3 | blast_opt | MS-TimeFilter | 0.052127 | 0.110418 | 0.044074 |
| C3 | main | MS-TimeFilter | 0.052966 | 0.112980 | 0.042585 |
| baseline_lstm_a4_96 | reference | LSTM | 0.051959 | 0.097959 | 0.033767 |
| baseline_raw_timefilter_a4_96 | reference | TimeFilter | 0.055817 | 0.174941 | 0.092875 |
