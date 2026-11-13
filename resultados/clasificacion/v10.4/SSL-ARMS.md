| Encoder (same reader, no ISIC rows) | spec | milk_dev | milk_test | pad_test | ham_holdout | ddi | fitz | bosque |
|---|---|---|---|---|---|---|---|---|
| shipped MoCo V10 s42 | cell:1:10 | 0.643 | 0.680 | 0.696 | 0.763 | 0.332 | 0.618 | 0.554 |
| shipped MoCo V10 s42 | cell:2:10 | 0.632 | 0.676 | 0.713 | 0.746 | 0.334 | 0.706 | 0.557 |
| SSL + clinical bucket, quota 4 | cell:1:10 | 0.636 | 0.673 | 0.701 | 0.774 | 0.318 | 0.622 | 0.566 |
| SSL + clinical bucket, quota 4 | cell:2:10 | 0.636 | 0.659 | 0.716 | 0.751 | 0.320 | 0.601 | 0.571 |
| SSL + clinical bucket, quota 8 | cell:1:10 | 0.646 | 0.658 | 0.720 | 0.763 | 0.306 | 0.617 | 0.563 |
| SSL + clinical bucket, quota 8 | cell:2:10 | 0.639 | 0.661 | 0.737 | 0.745 | 0.306 | 0.613 | 0.577 |
