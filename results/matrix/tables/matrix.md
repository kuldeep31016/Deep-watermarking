# Matrix results (matrix-20260918-025538-98a873)

n = 100 images from `/Users/kuldeepraj/deep-watermarking/data/processed/div2k_256/test`; blind columns come from the decoder trained for that exact point (blank = no such decoder).

| Bits | alpha | PSNR (dB) | SSIM | Non-blind acc | Blind acc | Blind exact | Blind decoder |
|---|---|---|---|---|---|---|---|
| 8 | 0.005 | 49.00 | 0.9986 | 0.9938 | 0.4788 | 0.00 | models/windowed_cnn_grid/8bit_a0.005/windowed_cnn_best.pt@e257e675d5b9 |
| 16 | 0.005 | 49.02 | 0.9985 | 0.9888 | 0.5700 | 0.00 | models/windowed_cnn_grid/16bit_a0.005/windowed_cnn_best.pt@78c5d254bc88 |
| 32 | 0.005 | 49.01 | 0.9985 | 0.9869 | 0.6031 | 0.00 | models/windowed_cnn_grid/32bit_a0.005/windowed_cnn_best.pt@1c3ef3c96e36 |
| 64 | 0.005 | 48.97 | 0.9985 | 0.9698 | 0.6445 | 0.00 | models/windowed_cnn_grid/64bit_a0.005/windowed_cnn_best.pt@639e33db8499 |
| 128 | 0.005 | 48.97 | 0.9985 | 0.8698 | 0.6199 | 0.00 | models/windowed_cnn_grid/128bit_a0.005/windowed_cnn_best.pt@69622428c2f8 |
| 256 | 0.005 | 48.95 | 0.9985 | 0.8703 |  |  |  |
| 8 | 0.010 | 45.70 | 0.9982 | 0.9962 | 0.5300 | 0.01 | models/windowed_cnn_grid/8bit_a0.010/windowed_cnn_best.pt@3ed281d175ee |
| 16 | 0.010 | 45.65 | 0.9978 | 0.9950 | 0.6019 | 0.00 | models/windowed_cnn_grid/16bit_a0.010/windowed_cnn_best.pt@af79e2a17126 |
| 32 | 0.010 | 45.60 | 0.9979 | 0.9819 | 0.6556 | 0.00 | models/windowed_cnn_grid/32bit_a0.010/windowed_cnn_best.pt@5fbf855c344d |
| 64 | 0.010 | 45.56 | 0.9980 | 0.9639 | 0.7008 | 0.00 | models/windowed_cnn_grid/64bit_a0.010/windowed_cnn_best.pt@426cd6b889d9 |
| 128 | 0.010 | 45.58 | 0.9980 | 0.8791 | 0.6723 | 0.00 | models/windowed_cnn_grid/128bit_a0.010/windowed_cnn_best.pt@69c296fe46a2 |
| 256 | 0.010 | 45.54 | 0.9980 | 0.8719 |  |  |  |
| 8 | 0.015 | 42.87 | 0.9977 | 0.9975 | 0.5537 | 0.00 | models/windowed_cnn_grid/8bit_a0.015/windowed_cnn_best.pt@231717150fda |
| 16 | 0.015 | 42.82 | 0.9971 | 0.9844 | 0.6431 | 0.00 | models/windowed_cnn_grid/16bit_a0.015/windowed_cnn_best.pt@30d8db51dac3 |
| 32 | 0.015 | 42.74 | 0.9972 | 0.9675 | 0.7109 | 0.00 | models/windowed_cnn_grid/32bit_a0.015/windowed_cnn_best.pt@e188a3e68949 |
| 64 | 0.015 | 42.70 | 0.9973 | 0.9347 | 0.7448 | 0.00 | models/windowed_cnn_grid/64bit_a0.015/windowed_cnn_best.pt@cf71fdb33c06 |
| 128 | 0.015 | 42.71 | 0.9974 | 0.8602 | 0.7091 | 0.00 | models/windowed_cnn_grid/128bit_a0.015/windowed_cnn_best.pt@63001e544de6 |
| 256 | 0.015 | 42.67 | 0.9972 | 0.8546 |  |  |  |
| 8 | 0.020 | 40.67 | 0.9970 | 0.9925 | 0.6112 | 0.04 | models/windowed_cnn_grid/8bit_a0.020/windowed_cnn_best.pt@2184f195afca |
| 16 | 0.020 | 40.60 | 0.9961 | 0.9719 | 0.6694 | 0.00 | models/windowed_cnn_grid/16bit_a0.020/windowed_cnn_best.pt@da4243c7ddca |
| 32 | 0.020 | 40.52 | 0.9963 | 0.9437 | 0.7462 | 0.00 | models/windowed_cnn_grid/32bit_a0.020/windowed_cnn_best.pt@1282bdd944b2 |
| 64 | 0.020 | 40.50 | 0.9964 | 0.9000 | 0.7648 | 0.00 | models/windowed_cnn_grid/64bit_a0.020/windowed_cnn_best.pt@ff2fedf06329 |
| 128 | 0.020 | 40.50 | 0.9965 | 0.8324 | 0.7230 | 0.00 | models/windowed_cnn_grid/128bit_a0.020/windowed_cnn_best.pt@185094eaf75f |
| 256 | 0.020 | 40.46 | 0.9963 | 0.8255 |  |  |  |
