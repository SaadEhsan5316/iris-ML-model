# Iris Classification with k-Nearest Neighbors (`mt`)

A small machine learning project where I explore the classic **Iris dataset** and train a **k-Nearest Neighbors (k-NN)** classifier to predict iris flower species from their measurements.

I built this as part of my AI and Data Science studies. My goal was to understand the full beginner workflow end to end: loading data, inspecting it, splitting it, visualizing it, training a model, and evaluating it.

---

## What This Project Does

1. **Loads** the Iris dataset from scikit-learn.
2. **Explores** its structure: keys, description, feature names, target names, array types and shapes.
3. **Splits** the data into training and test sets with `train_test_split` (`random_state=0` for reproducibility).
4. **Visualizes** the training data with a scatter matrix, colored by species, to see how the features relate.
5. **Trains** a `KNeighborsClassifier` with `n_neighbors=1`.
6. **Predicts** the species of a new, unseen flower measurement.
7. **Evaluates** the model on the test set, both manually (`np.mean(y_pred == y_test)`) and with `knn.score(...)`.

---

## About the Dataset

The Iris dataset has 150 samples across 3 species:

| Item | Details |
|------|---------|
| Classes | `setosa`, `versicolor`, `virginica` |
| Features | sepal length (cm), sepal width (cm), petal length (cm), petal width (cm) |
| Samples | 150 (split into a training set and a test set) |

---

## Tech Stack

- **Python** 3.14 or newer
- **scikit-learn**: dataset, train/test split, k-NN model
- **NumPy** and **pandas**: arrays and DataFrames
- **matplotlib** and **mglearn**: plotting and the scatter matrix colors
- **SciPy**, **IPython**: supporting libraries
- **FastAPI** and **psycopg**: listed in `pyproject.toml` for possible future use (an API and a database); the current script does not use them yet

---

## Getting Started

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) as the package and project manager

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd mt

# Install all dependencies from pyproject.toml
uv sync
```

### Running the Script

```bash
uv run python main.py
```

The script prints the dataset information and model results in the terminal, then opens a window with the scatter matrix plot at the end (`plt.show()`).

---

## What to Expect

- Printed details about the dataset (keys, target names, feature names, shapes).
- The shapes of the training and test splits.
- A scatter matrix of the four features, colored by species.
- A prediction for a new sample, `[[2, 4.6, 9, 0.5]]`, together with its predicted species name.
- Test set predictions and the final **test set accuracy**.

---

## Project Structure

```
mt/
├── main.py          # Iris exploration and k-NN classifier
├── pyproject.toml   # Project metadata and dependencies
└── README.md        # This file
```

---

## What I Learned

- Why we **split data** into training and test sets, and why a fixed `random_state` keeps results reproducible.
- Why it helps to **look at the data first**: a scatter matrix shows how well the classes separate.
- How k-NN makes predictions by comparing a new point to its closest training examples.
- Two ways to measure accuracy: computing it by hand and using `.score()`.
- That scikit-learn expects input as a **2D array** (`X_new` has shape `(1, 4)`, not `(4,)`).

---
## Ideas for Next Steps

- Try different values of `n_neighbors` (3, 5, 7) and compare test scores.
- Use cross-validation for a more reliable accuracy estimate.
- Compare k-NN with other classifiers such as logistic regression or decision trees.
- Wrap the trained model in a **FastAPI** endpoint so predictions can be requested over HTTP.
- Add a confusion matrix to see which species get mixed up.

---

## Author

**Saad Ehsan**
Undergraduate AI student, University of Agriculture, Faisalabad
GitHub: [SaadEhsan5316](https://github.com/SaadEhsan5316)
Email: saadehsan5316@gmail.com

I am still learning, so feedback and suggestions are very welcome!
