# Model architecture and design

## Pipeline
1. Load labeled images from class-named folders.
2. Resize images; apply task-reviewed augmentation to training data only.
3. Run three convolutional blocks, each with two 3×3 convolutions, batch normalization, ReLU, max pooling, and spatial dropout.
4. Global average pooling creates one feature vector per channel.
5. A dense classifier with dropout outputs class logits.
6. Cross-entropy loss trains the classifier; AdamW weight decay adds regularization.
7. Validation loss selects the best checkpoint and controls early stopping.

## Why regularization?
Medical image datasets may be small, imbalanced, and institution-specific. A high-capacity model may learn acquisition artifacts instead of robust anatomy or pathology-related patterns. Dropout, weight decay, augmentation, and early stopping can help, but none guarantees generalization.

## Evaluation
Do not report validation accuracy as final clinical performance. Evaluate on a patient-level held-out test set and, ideally, externally across sites and scanners. Include sensitivity and specificity per relevant class, precision, F1, ROC-AUC where appropriate, confidence intervals, calibration, subgroup analysis, and expert review of errors.

## Limitations
This starter has no bundled dataset or pretrained weights, and no DICOM handling, calibration, external validation, or clinical workflow integration. It must not be used for patient care.
