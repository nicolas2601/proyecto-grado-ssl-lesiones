# Additional architecture-bank comparisons

Descriptive development OOF results on the same approved cohorts. Do not mix this ranking with the controlled frozen-reader comparison.
Fine-tuning uses a distinct supervised training/epoch-selection procedure; CV repetitions are not three globally trained SSL encoders.
Frozen bank initializers/architectures differ; the timm DINOv2 bank is not the provenance-bound native reference in V5.
Full per-class precision/recall/F1 and support: BANK.csv. Source aggregate: bank-source.json.

| Model | Protocol | PAD macro-F1 | HAM macro-F1 |
|---|---|---:|---:|
| bank-alexnet-finetune | supervised fine-tuning | 0.5228 | 0.4680 |
| bank-alexnet-frozen | frozen architecture context | 0.4658 | 0.4702 |
| bank-dinov2-vitb14-finetune | supervised fine-tuning | 0.6128 | 0.4994 |
| bank-dinov2-vitb14-frozen | frozen architecture context | 0.5841 | 0.4914 |
| bank-dinov2-vits14-finetune | supervised fine-tuning | 0.5850 | 0.4480 |
| bank-dinov2-vits14-frozen | frozen architecture context | 0.5595 | 0.4759 |
| bank-efficientnet-b0-finetune | supervised fine-tuning | 0.3898 | 0.3782 |
| bank-efficientnet-b0-frozen | frozen architecture context | 0.5328 | 0.4367 |
| bank-resnet50-finetune | supervised fine-tuning | 0.5437 | 0.5092 |
| bank-resnet50-frozen | frozen architecture context | 0.4885 | 0.4156 |
| bank-simsiam-r50-finetune | supervised fine-tuning | 0.4382 | 0.3485 |
| bank-simsiam-r50-frozen | frozen architecture context | 0.5318 | 0.4573 |
| bank-vgg19-finetune | supervised fine-tuning | 0.5834 | 0.4822 |
| bank-vgg19-frozen | frozen architecture context | 0.4513 | 0.3862 |
| bank-vit-b16-finetune | supervised fine-tuning | 0.5731 | 0.5168 |
| bank-vit-b16-frozen | frozen architecture context | 0.5376 | 0.4715 |

No clinical winner or significance claim follows. Sealed test roles were not accessed.
