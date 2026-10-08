| Model | Val RMSE | Test RMSE (ms) | Test MAE (ms) | Test R² |
|---|---|---|---|---|
| Untrained model (predicts 0) - what the paper's test_model() effectively evaluates |  | 7459 | 4707 | -0.662 |
| Mean predictor (training mean) |  | 5821 | 4299 | -0.012 |
| Model A: Paper baseline (original pairing) | 5518 | 5818 | 4299 | -0.011 |
| Model B: Improved (corrected alignment) | 4264 | 5354 | 3645 | 0.144 |