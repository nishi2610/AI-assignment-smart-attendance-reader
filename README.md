# Smart Attendance Reader (Problem B): k-NN and PCA on handwritten digits

Individual assignment for *Introduction to Artificial Intelligence* (Dr. Paawan Sharma),
Pandit Deendayal Energy University.

**Author:** Nishi Purvinkumar Shah (25BIT051)

## What this project does
It tests whether a computer can read handwritten digits (0-9), as a first building block
for reading handwritten roll numbers, and whether all 64 pixel features are needed.

- **k-NN** (supervised) classifies digit images; k is chosen by 5-fold cross-validation.
- **PCA** (unsupervised) reduces the 64 pixel features; k-NN is then re-run on the reduced data.

Data: the 1,797 handwritten-digit images (8x8 pixels) that come with scikit-learn (`load_digits`).
No download is needed. Random seed 42 is used everywhere, so results are reproducible.

## How to run
```
pip install -r requirements.txt
python knn_pca_digits.py
```
The script prints the results and saves six figures into a `figures/` folder.

## Results (from my own run)
| Item | Result |
|---|---|
| Best k (5-fold CV) | 1 (CV accuracy 0.9756, tied with k = 5) |
| k-NN test accuracy | 96.67% (348 of 360) |
| PCA components for 90% / 95% / 99% variance | 31 / 40 / 54 |
| k-NN on 40 PCA components | 97.22% (350 of 360) |
