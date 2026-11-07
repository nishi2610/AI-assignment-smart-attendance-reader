"""
Smart Attendance Reader (Problem B)
Techniques used: k-NN (supervised) and PCA (unsupervised)

What this script does, in order:
  1. Loads handwritten digit pictures (0-9) that come built into scikit-learn.
  2. Splits them into a TRAINING part (to learn from) and a TEST part (to check).
  3. k-NN: finds the best k, then measures accuracy on the test part.
  4. PCA: shrinks each picture from 64 numbers to fewer numbers.
  5. Runs k-NN again on the shrunken pictures and compares.
  6. Saves every graph into a folder called "figures".

How to run:   pip install scikit-learn matplotlib
              python knn_pca_digits.py
"""

# ---------------------------------------------------------------- imports
import os                                    # to create the "figures" folder
import time                                  # to measure how long things take
import numpy as np                           # numbers and arrays
import matplotlib                            # drawing graphs
matplotlib.use("Agg")                        # draw to files, no pop-up window needed
import matplotlib.pyplot as plt              # the main graph-drawing tool

from sklearn.datasets import load_digits                      # the digit pictures
from sklearn.model_selection import train_test_split          # splits data in two parts
from sklearn.model_selection import cross_val_score           # tests a model several times
from sklearn.preprocessing import StandardScaler              # puts all pixels on the same scale
from sklearn.neighbors import KNeighborsClassifier            # the k-NN model
from sklearn.decomposition import PCA                         # the PCA tool
from sklearn.metrics import accuracy_score                    # % of correct answers
from sklearn.metrics import confusion_matrix                  # table of right/wrong answers
from sklearn.metrics import ConfusionMatrixDisplay            # draws that table
from sklearn.metrics import classification_report            # precision and recall per digit

SEED = 42                                    # fixed number so results are the same every run
os.makedirs("figures", exist_ok=True)        # make the "figures" folder if it does not exist

# ---------------------------------------------------------------- 1. load data
digits = load_digits()                       # load the dataset
X = digits.data                              # table: one row per picture, 64 columns (pixels)
y = digits.target                            # the correct digit (0-9) for every picture
print("Pictures:", X.shape[0], " Pixels per picture:", X.shape[1])

# Figure 1: show 2 example pictures of each digit
fig, axes = plt.subplots(2, 10, figsize=(10, 2.3))      # a grid of 2 rows x 10 columns
for digit in range(10):                                  # go through digits 0 to 9
    positions = np.where(y == digit)[0][:2]              # find the first 2 pictures of this digit
    for row, i in enumerate(positions):                  # draw each of them
        axes[row, digit].imshow(digits.images[i], cmap="gray_r")  # show the 8x8 picture
        axes[row, digit].axis("off")                     # hide the axis lines
plt.tight_layout()                                       # tidy spacing
plt.savefig("figures/fig1_samples.png", dpi=200)         # save the picture to a file
plt.close()                                              # close it so memory is freed

# ---------------------------------------------------------------- 2. split data
# 80% of pictures for training, 20% for testing.
# stratify=y keeps the same share of each digit in both parts.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print("Training pictures:", len(X_train), " Test pictures:", len(X_test))

# Scaling: give every pixel mean 0 and spread 1.
scaler = StandardScaler().fit(X_train)       # learn the scaling ONLY from training data
X_train_s = scaler.transform(X_train)        # apply to training data
X_test_s = scaler.transform(X_test)          # apply the same scaling to test data

# ---------------------------------------------------------------- 3. k-NN
# k-NN: to name a new picture, look at its k most similar training pictures
# and pick the digit that most of them have.
k_values = list(range(1, 16))                # try k = 1, 2, ... 15
cv_scores = []                               # will hold the score for each k
for k in k_values:                           # test every k
    model = KNeighborsClassifier(n_neighbors=k)                   # k-NN with this k
    score = cross_val_score(model, X_train_s, y_train, cv=5).mean()  # 5-fold cross-validation
    cv_scores.append(score)                  # remember the score
best_k = k_values[int(np.argmax(cv_scores))] # the k with the highest score
print("Best k:", best_k, " (cross-validated accuracy %.4f)" % max(cv_scores))

# Figure 2: accuracy for each k
plt.figure(figsize=(5.5, 3.2))
plt.plot(k_values, cv_scores, "o-")          # line with dots
plt.axvline(best_k, ls="--", c="gray")       # dashed line at the best k
plt.xlabel("k (number of neighbours)")
plt.ylabel("Cross-validated accuracy")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig2_knn_k.png", dpi=200)
plt.close()

# Train the final k-NN and test it ONCE on the test pictures.
knn = KNeighborsClassifier(n_neighbors=best_k)   # k-NN with the best k
knn.fit(X_train_s, y_train)                      # "training" = just storing the pictures
start = time.time()                              # start the stopwatch
pred = knn.predict(X_test_s)                     # guess the digit of every test picture
knn_time = time.time() - start                   # time taken
knn_acc = accuracy_score(y_test, pred)           # share of correct guesses
print("k-NN test accuracy: %.4f  (%d of %d correct)" %
      (knn_acc, int(knn_acc * len(y_test)), len(y_test)))

# Confusion matrix: rows = true digit, columns = predicted digit.
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False, cmap="Blues")
plt.tight_layout()
plt.savefig("figures/fig3_knn_confusion.png", dpi=200)
plt.close()

# Which digits are the hardest? (recall = how many of each digit were found)
report = classification_report(y_test, pred, output_dict=True)
for d in range(10):
    print("  digit %d: recall %.3f" % (d, report[str(d)]["recall"]))

# ---------------------------------------------------------------- 4. PCA
# PCA: finds new "directions" that hold the most information, so we can keep
# only the first few of them instead of all 64 pixels.
pca_full = PCA().fit(X_train_s)                          # PCA keeping all 64 directions
cumulative = np.cumsum(pca_full.explained_variance_ratio_)   # running total of information
n90 = int(np.argmax(cumulative >= 0.90) + 1)             # how many parts give 90% of the information
n95 = int(np.argmax(cumulative >= 0.95) + 1)             # ... 95%
n99 = int(np.argmax(cumulative >= 0.99) + 1)             # ... 99%
print("Components needed for 90%%, 95%%, 99%% of the information: %d, %d, %d" % (n90, n95, n99))
print("PC1 holds %.1f%%, PC2 holds %.1f%%" %
      (100 * pca_full.explained_variance_ratio_[0], 100 * pca_full.explained_variance_ratio_[1]))

# Figure 4: how much information we keep as we add components
plt.figure(figsize=(5.5, 3.2))
plt.plot(range(1, 65), cumulative)
plt.axhline(0.95, ls="--", c="gray")         # the 95% line
plt.axvline(n95, ls="--", c="gray")          # where we reach 95%
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig4_pca_variance.png", dpi=200)
plt.close()

# Figure 5: every training picture drawn as one dot, using only 2 components
pca_2d = PCA(n_components=2).fit(X_train_s)              # PCA with only 2 directions
points = pca_2d.transform(X_train_s)                     # each picture becomes 2 numbers
plt.figure(figsize=(5.8, 4.6))
scatter = plt.scatter(points[:, 0], points[:, 1], c=y_train, cmap="tab10", s=10)  # colour = digit
plt.legend(*scatter.legend_elements(), title="Digit", loc="center left",
           bbox_to_anchor=(1, 0.5), fontsize=8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("figures/fig5_pca_2d.png", dpi=200)
plt.close()

# ---------------------------------------------------------------- 5. k-NN on PCA data
# Does k-NN still work if we keep only the first n components?
counts = [2, 5, 10, 20, n95, 64]             # the numbers of components we will try
accuracies = []                              # accuracy for each choice
for n in counts:
    pca = PCA(n_components=n).fit(X_train_s)              # PCA keeping n components
    A = pca.transform(X_train_s)                          # shrink the training pictures
    B = pca.transform(X_test_s)                           # shrink the test pictures (same PCA)
    m = KNeighborsClassifier(n_neighbors=best_k).fit(A, y_train)  # k-NN on the shrunken data
    acc = accuracy_score(y_test, m.predict(B))            # test accuracy
    accuracies.append(acc)
    print("k-NN with %2d components: accuracy %.4f" % (n, acc))

# Figure 6: accuracy against number of components
plt.figure(figsize=(5.5, 3.2))
plt.plot(counts, accuracies, "o-")
plt.xlabel("Number of PCA components")
plt.ylabel("k-NN test accuracy")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig6_knn_vs_components.png", dpi=200)
plt.close()

print("Done. Figures are in the 'figures' folder.")
