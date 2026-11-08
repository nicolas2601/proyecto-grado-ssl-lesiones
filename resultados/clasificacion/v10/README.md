# V10 lesion classification OOF results

Development-only results; these tables contain aggregate metrics, not image-level predictions.

source_sha256=b0df0e23a998a405d06d880d389df9ad84f0977f3fc0c348419856372c30e7b1
protocol_sha256=1144e857e768a94e4e8df5129e5ec0b5f9f73d3f5c2a5f2982b0e42e9e335414

## PAD6 and HAM7 results

## PAD

| Method | Accuracy | Macro-F1 | Balanced accuracy | Macro OVR AUC | ECE | Trained SSL seeds | Encoder checkpoints | CV repetitions per encoder |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MoCo quota4 CLS4 (proposed) | 0.7209 | 0.6746 | 0.6878 | 0.9194 | 0.0338 | 3 | 3 | 3 |
| MoCo M0 CLS | 0.7018 | 0.6438 | 0.6543 | 0.9046 | 0.0508 | 3 | 3 | 3 |
| MoCo M0 CLS4 | 0.7082 | 0.6603 | 0.6747 | 0.9141 | 0.0429 | 3 | 3 | 3 |
| MoCo M0 CLS plus patch mean | 0.7055 | 0.6429 | 0.6507 | 0.9068 | 0.0504 | 3 | 3 | 3 |
| MoCo quota4 CLS | 0.7104 | 0.6630 | 0.6760 | 0.9092 | 0.0510 | 3 | 3 | 3 |
| MoCo quota4 CLS plus patch mean | 0.7104 | 0.6554 | 0.6654 | 0.9133 | 0.0468 | 3 | 3 | 3 |
| MoCo quota4 50% weight interpolation | 0.7048 | 0.6560 | 0.6629 | 0.9077 | 0.0544 | 3 | 3 | 3 |
| MoCo M0 EMA teacher | 0.6961 | 0.6393 | 0.6510 | 0.9057 | 0.0444 | 3 | 3 | 3 |
| MoCo quota4 EMA teacher | 0.7069 | 0.6580 | 0.6718 | 0.9099 | 0.0452 | 3 | 3 | 3 |
| Public DINOv2 (CLS) | 0.6514 | 0.5868 | 0.5961 | 0.8736 | 0.0741 | 0 | 1 | 3 |
| Legacy baseline (method identity unverified) | 0.6558 | 0.5973 | 0.6091 | 0.8810 | 0.0577 | 3 | 3 | 3 |
| VICReg | 0.6923 | 0.6381 | 0.6457 | 0.9040 | 0.0496 | 3 | 3 | 3 |
| Barlow Twins + VICReg | 0.6872 | 0.6369 | 0.6514 | 0.9003 | 0.0462 | 3 | 3 | 3 |
| SimSiam R50 (frozen bank; architecture context) | 0.5982 | 0.5318 | 0.5381 | 0.8423 | 0.0578 | 0 | 1 | 3 |

| Method | Class | Precision | Recall | F1 | Images | Components/groups |
|---|---|---:|---:|---:|---:|---:|
| MoCo quota4 CLS4 (proposed) | ACK | 0.8027 | 0.8076 | 0.8051 | 362 | 258 |
| MoCo quota4 CLS4 (proposed) | BCC | 0.7730 | 0.7020 | 0.7356 | 418 | 247 |
| MoCo quota4 CLS4 (proposed) | MEL | 0.6526 | 0.7325 | 0.6901 | 27 | 18 |
| MoCo quota4 CLS4 (proposed) | NEV | 0.7604 | 0.7941 | 0.7766 | 116 | 87 |
| MoCo quota4 CLS4 (proposed) | SCC | 0.2584 | 0.3193 | 0.2854 | 95 | 67 |
| MoCo quota4 CLS4 (proposed) | SEK | 0.7398 | 0.7711 | 0.7549 | 116 | 87 |
| MoCo M0 CLS | ACK | 0.7863 | 0.7904 | 0.7883 | 362 | 258 |
| MoCo M0 CLS | BCC | 0.7646 | 0.6850 | 0.7225 | 418 | 247 |
| MoCo M0 CLS | MEL | 0.5828 | 0.6091 | 0.5936 | 27 | 18 |
| MoCo M0 CLS | NEV | 0.7247 | 0.7730 | 0.7478 | 116 | 87 |
| MoCo M0 CLS | SCC | 0.2439 | 0.3135 | 0.2737 | 95 | 67 |
| MoCo M0 CLS | SEK | 0.7197 | 0.7548 | 0.7367 | 116 | 87 |
| MoCo M0 CLS4 | ACK | 0.7938 | 0.7950 | 0.7943 | 362 | 258 |
| MoCo M0 CLS4 | BCC | 0.7702 | 0.6858 | 0.7255 | 418 | 247 |
| MoCo M0 CLS4 | MEL | 0.6262 | 0.7037 | 0.6611 | 27 | 18 |
| MoCo M0 CLS4 | NEV | 0.7602 | 0.7826 | 0.7711 | 116 | 87 |
| MoCo M0 CLS4 | SCC | 0.2374 | 0.3099 | 0.2684 | 95 | 67 |
| MoCo M0 CLS4 | SEK | 0.7145 | 0.7711 | 0.7416 | 116 | 87 |
| MoCo M0 CLS plus patch mean | ACK | 0.7864 | 0.7864 | 0.7864 | 362 | 258 |
| MoCo M0 CLS plus patch mean | BCC | 0.7633 | 0.7012 | 0.7308 | 418 | 247 |
| MoCo M0 CLS plus patch mean | MEL | 0.5886 | 0.5844 | 0.5852 | 27 | 18 |
| MoCo M0 CLS plus patch mean | NEV | 0.7220 | 0.7711 | 0.7451 | 116 | 87 |
| MoCo M0 CLS plus patch mean | SCC | 0.2429 | 0.2889 | 0.2634 | 95 | 67 |
| MoCo M0 CLS plus patch mean | SEK | 0.7225 | 0.7720 | 0.7463 | 116 | 87 |
| MoCo quota4 CLS | ACK | 0.7928 | 0.7980 | 0.7953 | 362 | 258 |
| MoCo quota4 CLS | BCC | 0.7628 | 0.6885 | 0.7236 | 418 | 247 |
| MoCo quota4 CLS | MEL | 0.6357 | 0.6996 | 0.6642 | 27 | 18 |
| MoCo quota4 CLS | NEV | 0.7176 | 0.7711 | 0.7433 | 116 | 87 |
| MoCo quota4 CLS | SCC | 0.2636 | 0.3287 | 0.2920 | 95 | 67 |
| MoCo quota4 CLS | SEK | 0.7495 | 0.7701 | 0.7595 | 116 | 87 |
| MoCo quota4 CLS plus patch mean | ACK | 0.7963 | 0.7999 | 0.7981 | 362 | 258 |
| MoCo quota4 CLS plus patch mean | BCC | 0.7625 | 0.6980 | 0.7287 | 418 | 247 |
| MoCo quota4 CLS plus patch mean | MEL | 0.6270 | 0.6626 | 0.6433 | 27 | 18 |
| MoCo quota4 CLS plus patch mean | NEV | 0.7170 | 0.7711 | 0.7430 | 116 | 87 |
| MoCo quota4 CLS plus patch mean | SCC | 0.2462 | 0.2947 | 0.2680 | 95 | 67 |
| MoCo quota4 CLS plus patch mean | SEK | 0.7368 | 0.7663 | 0.7512 | 116 | 87 |
| MoCo quota4 50% weight interpolation | ACK | 0.7848 | 0.7977 | 0.7912 | 362 | 258 |
| MoCo quota4 50% weight interpolation | BCC | 0.7494 | 0.6834 | 0.7147 | 418 | 247 |
| MoCo quota4 50% weight interpolation | MEL | 0.6578 | 0.6584 | 0.6564 | 27 | 18 |
| MoCo quota4 50% weight interpolation | NEV | 0.7288 | 0.7692 | 0.7483 | 116 | 87 |
| MoCo quota4 50% weight interpolation | SCC | 0.2456 | 0.2959 | 0.2681 | 95 | 67 |
| MoCo quota4 50% weight interpolation | SEK | 0.7419 | 0.7730 | 0.7571 | 116 | 87 |
| MoCo M0 EMA teacher | ACK | 0.7837 | 0.7848 | 0.7842 | 362 | 258 |
| MoCo M0 EMA teacher | BCC | 0.7628 | 0.6746 | 0.7160 | 418 | 247 |
| MoCo M0 EMA teacher | MEL | 0.5772 | 0.6049 | 0.5893 | 27 | 18 |
| MoCo M0 EMA teacher | NEV | 0.7090 | 0.7663 | 0.7362 | 116 | 87 |
| MoCo M0 EMA teacher | SCC | 0.2422 | 0.3216 | 0.2759 | 95 | 67 |
| MoCo M0 EMA teacher | SEK | 0.7162 | 0.7538 | 0.7344 | 116 | 87 |
| MoCo quota4 EMA teacher | ACK | 0.7951 | 0.7916 | 0.7933 | 362 | 258 |
| MoCo quota4 EMA teacher | BCC | 0.7603 | 0.6874 | 0.7219 | 418 | 247 |
| MoCo quota4 EMA teacher | MEL | 0.6218 | 0.6914 | 0.6537 | 27 | 18 |
| MoCo quota4 EMA teacher | NEV | 0.7182 | 0.7692 | 0.7426 | 116 | 87 |
| MoCo quota4 EMA teacher | SCC | 0.2540 | 0.3193 | 0.2823 | 95 | 67 |
| MoCo quota4 EMA teacher | SEK | 0.7374 | 0.7720 | 0.7541 | 116 | 87 |
| Public DINOv2 (CLS) | ACK | 0.7611 | 0.7449 | 0.7529 | 362 | 258 |
| Public DINOv2 (CLS) | BCC | 0.6917 | 0.6443 | 0.6671 | 418 | 247 |
| Public DINOv2 (CLS) | MEL | 0.4926 | 0.5432 | 0.5163 | 27 | 18 |
| Public DINOv2 (CLS) | NEV | 0.6821 | 0.7270 | 0.7038 | 116 | 87 |
| Public DINOv2 (CLS) | SCC | 0.2182 | 0.2737 | 0.2427 | 95 | 67 |
| Public DINOv2 (CLS) | SEK | 0.6330 | 0.6437 | 0.6381 | 116 | 87 |
| Legacy baseline (method identity unverified) | ACK | 0.7669 | 0.7483 | 0.7575 | 362 | 258 |
| Legacy baseline (method identity unverified) | BCC | 0.6965 | 0.6396 | 0.6668 | 418 | 247 |
| Legacy baseline (method identity unverified) | MEL | 0.5101 | 0.5761 | 0.5400 | 27 | 18 |
| Legacy baseline (method identity unverified) | NEV | 0.6907 | 0.7356 | 0.7124 | 116 | 87 |
| Legacy baseline (method identity unverified) | SCC | 0.2273 | 0.2947 | 0.2565 | 95 | 67 |
| Legacy baseline (method identity unverified) | SEK | 0.6422 | 0.6600 | 0.6509 | 116 | 87 |
| VICReg | ACK | 0.7775 | 0.7855 | 0.7814 | 362 | 258 |
| VICReg | BCC | 0.7647 | 0.6736 | 0.7161 | 418 | 247 |
| VICReg | MEL | 0.6319 | 0.6091 | 0.6184 | 27 | 18 |
| VICReg | NEV | 0.7029 | 0.7711 | 0.7349 | 116 | 87 |
| VICReg | SCC | 0.2408 | 0.3205 | 0.2749 | 95 | 67 |
| VICReg | SEK | 0.6919 | 0.7146 | 0.7029 | 116 | 87 |
| Barlow Twins + VICReg | ACK | 0.7779 | 0.7848 | 0.7813 | 362 | 258 |
| Barlow Twins + VICReg | BCC | 0.7670 | 0.6547 | 0.7062 | 418 | 247 |
| Barlow Twins + VICReg | MEL | 0.6164 | 0.6502 | 0.6324 | 27 | 18 |
| Barlow Twins + VICReg | NEV | 0.6921 | 0.7826 | 0.7344 | 116 | 87 |
| Barlow Twins + VICReg | SCC | 0.2400 | 0.3322 | 0.2784 | 95 | 67 |
| Barlow Twins + VICReg | SEK | 0.6753 | 0.7040 | 0.6891 | 116 | 87 |
| SimSiam R50 (frozen bank; architecture context) | ACK | 0.6877 | 0.7136 | 0.7004 | 362 | 258 |
| SimSiam R50 (frozen bank; architecture context) | BCC | 0.6563 | 0.5742 | 0.6125 | 418 | 247 |
| SimSiam R50 (frozen bank; architecture context) | MEL | 0.4800 | 0.4444 | 0.4614 | 27 | 18 |
| SimSiam R50 (frozen bank; architecture context) | NEV | 0.6429 | 0.7126 | 0.6759 | 116 | 87 |
| SimSiam R50 (frozen bank; architecture context) | SCC | 0.2206 | 0.2667 | 0.2414 | 95 | 67 |
| SimSiam R50 (frozen bank; architecture context) | SEK | 0.4829 | 0.5172 | 0.4994 | 116 | 87 |

### Nominal class-level leaders

| Class | Best nominal F1 | Best nominal recall |
|---|---|---|
| ACK | MoCo quota4 CLS4 (proposed) (0.8051) | MoCo quota4 CLS4 (proposed) (0.8076) |
| BCC | MoCo quota4 CLS4 (proposed) (0.7356) | MoCo quota4 CLS4 (proposed) (0.7020) |
| MEL | MoCo quota4 CLS4 (proposed) (0.6901) | MoCo quota4 CLS4 (proposed) (0.7325) |
| NEV | MoCo quota4 CLS4 (proposed) (0.7766) | MoCo quota4 CLS4 (proposed) (0.7941) |
| SCC | MoCo quota4 CLS (0.2920) | Barlow Twins + VICReg (0.3322) |
| SEK | MoCo quota4 CLS (0.7595) | MoCo quota4 50% weight interpolation (0.7730) |

Support counts are cohort totals; group/component counts can overlap across classes. 
Trained SSL seeds counts only reported seed IDs; zero with one checkpoint means the seed ID is 
not present in this evidence bundle. Leaders are exact descriptive maxima; exact ties list all method 
names alphabetically. No statistical significance is implied.

Proposed MoCo result per row describes the corresponding development OOF readouts; it is not a test-set result, ensemble, or single globally fitted deployment reader.


## HAM

| Method | Accuracy | Macro-F1 | Balanced accuracy | Macro OVR AUC | ECE | Trained SSL seeds | Encoder checkpoints | CV repetitions per encoder |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MoCo quota4 CLS4 (proposed) | 0.7608 | 0.6205 | 0.6358 | 0.9367 | 0.0351 | 3 | 3 | 3 |
| MoCo M0 CLS | 0.7456 | 0.6020 | 0.6315 | 0.9298 | 0.0418 | 3 | 3 | 3 |
| MoCo M0 CLS4 | 0.7569 | 0.6114 | 0.6303 | 0.9357 | 0.0341 | 3 | 3 | 3 |
| MoCo M0 CLS plus patch mean | 0.7570 | 0.5964 | 0.6228 | 0.9331 | 0.0501 | 3 | 3 | 3 |
| MoCo quota4 CLS | 0.7479 | 0.6017 | 0.6332 | 0.9310 | 0.0445 | 3 | 3 | 3 |
| MoCo quota4 CLS plus patch mean | 0.7596 | 0.6086 | 0.6355 | 0.9356 | 0.0464 | 3 | 3 | 3 |
| MoCo quota4 50% weight interpolation | 0.7440 | 0.5827 | 0.6133 | 0.9283 | 0.0506 | 3 | 3 | 3 |
| MoCo M0 EMA teacher | 0.7424 | 0.5982 | 0.6244 | 0.9285 | 0.0430 | 3 | 3 | 3 |
| MoCo quota4 EMA teacher | 0.7491 | 0.6028 | 0.6320 | 0.9304 | 0.0521 | 3 | 3 | 3 |
| Public DINOv2 (CLS) | 0.7064 | 0.5049 | 0.5325 | 0.8994 | 0.0361 | 0 | 1 | 3 |
| Legacy baseline (method identity unverified) | 0.7103 | 0.5179 | 0.5420 | 0.9063 | 0.0469 | 3 | 3 | 3 |
| VICReg | 0.7308 | 0.5461 | 0.5686 | 0.9216 | 0.0755 | 3 | 3 | 3 |
| Barlow Twins + VICReg | 0.7152 | 0.5215 | 0.5507 | 0.9162 | 0.0840 | 3 | 3 | 3 |
| SimSiam R50 (frozen bank; architecture context) | 0.6996 | 0.4573 | 0.4754 | 0.8920 | 0.0325 | 0 | 1 | 3 |

| Method | Class | Precision | Recall | F1 | Images | Components/groups |
|---|---|---:|---:|---:|---:|---:|
| MoCo quota4 CLS4 (proposed) | akiec | 0.3960 | 0.4509 | 0.4210 | 52 | 36 |
| MoCo quota4 CLS4 (proposed) | bcc | 0.6578 | 0.7764 | 0.7120 | 82 | 57 |
| MoCo quota4 CLS4 (proposed) | bkl | 0.5298 | 0.6159 | 0.5696 | 173 | 122 |
| MoCo quota4 CLS4 (proposed) | df | 0.6522 | 0.3580 | 0.4602 | 18 | 6 |
| MoCo quota4 CLS4 (proposed) | mel | 0.4155 | 0.5385 | 0.4689 | 169 | 98 |
| MoCo quota4 CLS4 (proposed) | nv | 0.9293 | 0.8422 | 0.8835 | 1012 | 816 |
| MoCo quota4 CLS4 (proposed) | vasc | 0.7997 | 0.8687 | 0.8286 | 22 | 17 |
| MoCo M0 CLS | akiec | 0.3337 | 0.4145 | 0.3693 | 52 | 36 |
| MoCo M0 CLS | bcc | 0.6825 | 0.8130 | 0.7418 | 82 | 57 |
| MoCo M0 CLS | bkl | 0.4866 | 0.5742 | 0.5266 | 173 | 122 |
| MoCo M0 CLS | df | 0.6059 | 0.3951 | 0.4767 | 18 | 6 |
| MoCo M0 CLS | mel | 0.4003 | 0.5233 | 0.4534 | 169 | 98 |
| MoCo M0 CLS | nv | 0.9297 | 0.8270 | 0.8753 | 1012 | 816 |
| MoCo M0 CLS | vasc | 0.6935 | 0.8737 | 0.7710 | 22 | 17 |
| MoCo M0 CLS4 | akiec | 0.3926 | 0.4615 | 0.4239 | 52 | 36 |
| MoCo M0 CLS4 | bcc | 0.6722 | 0.7724 | 0.7180 | 82 | 57 |
| MoCo M0 CLS4 | bkl | 0.5215 | 0.6224 | 0.5674 | 173 | 122 |
| MoCo M0 CLS4 | df | 0.6281 | 0.3457 | 0.4433 | 18 | 6 |
| MoCo M0 CLS4 | mel | 0.4121 | 0.5293 | 0.4632 | 169 | 98 |
| MoCo M0 CLS4 | nv | 0.9280 | 0.8373 | 0.8802 | 1012 | 816 |
| MoCo M0 CLS4 | vasc | 0.7349 | 0.8434 | 0.7839 | 22 | 17 |
| MoCo M0 CLS plus patch mean | akiec | 0.3807 | 0.4530 | 0.4130 | 52 | 36 |
| MoCo M0 CLS plus patch mean | bcc | 0.6776 | 0.7778 | 0.7238 | 82 | 57 |
| MoCo M0 CLS plus patch mean | bkl | 0.5313 | 0.6012 | 0.5639 | 173 | 122 |
| MoCo M0 CLS plus patch mean | df | 0.5563 | 0.3025 | 0.3901 | 18 | 6 |
| MoCo M0 CLS plus patch mean | mel | 0.4131 | 0.5122 | 0.4571 | 169 | 98 |
| MoCo M0 CLS plus patch mean | nv | 0.9217 | 0.8441 | 0.8811 | 1012 | 816 |
| MoCo M0 CLS plus patch mean | vasc | 0.6556 | 0.8687 | 0.7460 | 22 | 17 |
| MoCo quota4 CLS | akiec | 0.3487 | 0.4487 | 0.3918 | 52 | 36 |
| MoCo quota4 CLS | bcc | 0.6564 | 0.7967 | 0.7194 | 82 | 57 |
| MoCo quota4 CLS | bkl | 0.4995 | 0.5678 | 0.5312 | 173 | 122 |
| MoCo quota4 CLS | df | 0.5789 | 0.3827 | 0.4562 | 18 | 6 |
| MoCo quota4 CLS | mel | 0.3971 | 0.5266 | 0.4525 | 169 | 98 |
| MoCo quota4 CLS | nv | 0.9335 | 0.8307 | 0.8790 | 1012 | 816 |
| MoCo quota4 CLS | vasc | 0.7050 | 0.8788 | 0.7814 | 22 | 17 |
| MoCo quota4 CLS plus patch mean | akiec | 0.3775 | 0.4487 | 0.4098 | 52 | 36 |
| MoCo quota4 CLS plus patch mean | bcc | 0.6760 | 0.7873 | 0.7270 | 82 | 57 |
| MoCo quota4 CLS plus patch mean | bkl | 0.5276 | 0.6044 | 0.5633 | 173 | 122 |
| MoCo quota4 CLS plus patch mean | df | 0.5801 | 0.3642 | 0.4432 | 18 | 6 |
| MoCo quota4 CLS plus patch mean | mel | 0.4140 | 0.5155 | 0.4590 | 169 | 98 |
| MoCo quota4 CLS plus patch mean | nv | 0.9272 | 0.8450 | 0.8841 | 1012 | 816 |
| MoCo quota4 CLS plus patch mean | vasc | 0.6898 | 0.8838 | 0.7737 | 22 | 17 |
| MoCo quota4 50% weight interpolation | akiec | 0.3074 | 0.3739 | 0.3372 | 52 | 36 |
| MoCo quota4 50% weight interpolation | bcc | 0.6271 | 0.7385 | 0.6778 | 82 | 57 |
| MoCo quota4 50% weight interpolation | bkl | 0.5142 | 0.5716 | 0.5413 | 173 | 122 |
| MoCo quota4 50% weight interpolation | df | 0.5015 | 0.3704 | 0.4237 | 18 | 6 |
| MoCo quota4 50% weight interpolation | mel | 0.3906 | 0.5273 | 0.4484 | 169 | 98 |
| MoCo quota4 50% weight interpolation | nv | 0.9305 | 0.8328 | 0.8788 | 1012 | 816 |
| MoCo quota4 50% weight interpolation | vasc | 0.6904 | 0.8788 | 0.7716 | 22 | 17 |
| MoCo M0 EMA teacher | akiec | 0.3385 | 0.4103 | 0.3704 | 52 | 36 |
| MoCo M0 EMA teacher | bcc | 0.6747 | 0.8022 | 0.7327 | 82 | 57 |
| MoCo M0 EMA teacher | bkl | 0.4782 | 0.5697 | 0.5199 | 173 | 122 |
| MoCo M0 EMA teacher | df | 0.6431 | 0.3889 | 0.4827 | 18 | 6 |
| MoCo M0 EMA teacher | mel | 0.3925 | 0.5155 | 0.4455 | 169 | 98 |
| MoCo M0 EMA teacher | nv | 0.9282 | 0.8258 | 0.8740 | 1012 | 816 |
| MoCo M0 EMA teacher | vasc | 0.6888 | 0.8586 | 0.7621 | 22 | 17 |
| MoCo quota4 EMA teacher | akiec | 0.3552 | 0.4466 | 0.3950 | 52 | 36 |
| MoCo quota4 EMA teacher | bcc | 0.6546 | 0.7940 | 0.7172 | 82 | 57 |
| MoCo quota4 EMA teacher | bkl | 0.5006 | 0.5652 | 0.5307 | 173 | 122 |
| MoCo quota4 EMA teacher | df | 0.5980 | 0.3951 | 0.4716 | 18 | 6 |
| MoCo quota4 EMA teacher | mel | 0.3974 | 0.5141 | 0.4482 | 169 | 98 |
| MoCo quota4 EMA teacher | nv | 0.9301 | 0.8353 | 0.8801 | 1012 | 816 |
| MoCo quota4 EMA teacher | vasc | 0.7021 | 0.8737 | 0.7768 | 22 | 17 |
| Public DINOv2 (CLS) | akiec | 0.3213 | 0.3654 | 0.3415 | 52 | 36 |
| Public DINOv2 (CLS) | bcc | 0.4658 | 0.5732 | 0.5139 | 82 | 57 |
| Public DINOv2 (CLS) | bkl | 0.4436 | 0.5337 | 0.4844 | 173 | 122 |
| Public DINOv2 (CLS) | df | 0.3599 | 0.1667 | 0.2240 | 18 | 6 |
| Public DINOv2 (CLS) | mel | 0.3530 | 0.4931 | 0.4114 | 169 | 98 |
| Public DINOv2 (CLS) | nv | 0.9221 | 0.8076 | 0.8611 | 1012 | 816 |
| Public DINOv2 (CLS) | vasc | 0.6267 | 0.7879 | 0.6978 | 22 | 17 |
| Legacy baseline (method identity unverified) | akiec | 0.3062 | 0.3333 | 0.3188 | 52 | 36 |
| Legacy baseline (method identity unverified) | bcc | 0.4977 | 0.5976 | 0.5430 | 82 | 57 |
| Legacy baseline (method identity unverified) | bkl | 0.4596 | 0.5517 | 0.5014 | 173 | 122 |
| Legacy baseline (method identity unverified) | df | 0.4815 | 0.2284 | 0.3046 | 18 | 6 |
| Legacy baseline (method identity unverified) | mel | 0.3397 | 0.4845 | 0.3993 | 169 | 98 |
| Legacy baseline (method identity unverified) | nv | 0.9253 | 0.8105 | 0.8641 | 1012 | 816 |
| Legacy baseline (method identity unverified) | vasc | 0.6213 | 0.7879 | 0.6942 | 22 | 17 |
| VICReg | akiec | 0.3534 | 0.4402 | 0.3913 | 52 | 36 |
| VICReg | bcc | 0.6287 | 0.6734 | 0.6496 | 82 | 57 |
| VICReg | bkl | 0.4676 | 0.5247 | 0.4944 | 173 | 122 |
| VICReg | df | 0.4779 | 0.3025 | 0.3661 | 18 | 6 |
| VICReg | mel | 0.3884 | 0.4938 | 0.4346 | 169 | 98 |
| VICReg | nv | 0.9111 | 0.8332 | 0.8704 | 1012 | 816 |
| VICReg | vasc | 0.5444 | 0.7121 | 0.6165 | 22 | 17 |
| Barlow Twins + VICReg | akiec | 0.3574 | 0.4594 | 0.4015 | 52 | 36 |
| Barlow Twins + VICReg | bcc | 0.5800 | 0.6802 | 0.6260 | 82 | 57 |
| Barlow Twins + VICReg | bkl | 0.4531 | 0.5228 | 0.4854 | 173 | 122 |
| Barlow Twins + VICReg | df | 0.3757 | 0.2716 | 0.3145 | 18 | 6 |
| Barlow Twins + VICReg | mel | 0.3673 | 0.4997 | 0.4233 | 169 | 98 |
| Barlow Twins + VICReg | nv | 0.9169 | 0.8103 | 0.8603 | 1012 | 816 |
| Barlow Twins + VICReg | vasc | 0.4836 | 0.6111 | 0.5392 | 22 | 17 |
| SimSiam R50 (frozen bank; architecture context) | akiec | 0.3241 | 0.3974 | 0.3569 | 52 | 36 |
| SimSiam R50 (frozen bank; architecture context) | bcc | 0.4538 | 0.5813 | 0.5095 | 82 | 57 |
| SimSiam R50 (frozen bank; architecture context) | bkl | 0.4191 | 0.5029 | 0.4571 | 173 | 122 |
| SimSiam R50 (frozen bank; architecture context) | df | 0.3150 | 0.1481 | 0.1978 | 18 | 6 |
| SimSiam R50 (frozen bank; architecture context) | mel | 0.3495 | 0.4418 | 0.3901 | 169 | 98 |
| SimSiam R50 (frozen bank; architecture context) | nv | 0.9135 | 0.8169 | 0.8624 | 1012 | 816 |
| SimSiam R50 (frozen bank; architecture context) | vasc | 0.4155 | 0.4394 | 0.4271 | 22 | 17 |

### Nominal class-level leaders

| Class | Best nominal F1 | Best nominal recall |
|---|---|---|
| akiec | MoCo M0 CLS4 (0.4239) | MoCo M0 CLS4 (0.4615) |
| bcc | MoCo M0 CLS (0.7418) | MoCo M0 CLS (0.8130) |
| bkl | MoCo quota4 CLS4 (proposed) (0.5696) | MoCo M0 CLS4 (0.6224) |
| df | MoCo M0 EMA teacher (0.4827) | MoCo M0 CLS, MoCo quota4 EMA teacher (0.3951) |
| mel | MoCo quota4 CLS4 (proposed) (0.4689) | MoCo quota4 CLS4 (proposed) (0.5385) |
| nv | MoCo quota4 CLS plus patch mean (0.8841) | MoCo quota4 CLS plus patch mean (0.8450) |
| vasc | MoCo quota4 CLS4 (proposed) (0.8286) | MoCo quota4 CLS plus patch mean (0.8838) |

Support counts are cohort totals; group/component counts can overlap across classes. 
Trained SSL seeds counts only reported seed IDs; zero with one checkpoint means the seed ID is 
not present in this evidence bundle. Leaders are exact descriptive maxima; exact ties list all method 
names alphabetically. No statistical significance is implied.

Proposed MoCo result per row describes the corresponding development OOF readouts; it is not a test-set result, ensemble, or single globally fitted deployment reader.
