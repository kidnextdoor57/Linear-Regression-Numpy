# Linear Regression from Scratch with NumPy

A complete implementation of Linear Regression built from the ground up using only NumPy — no scikit-learn, no shortcuts. The custom model is benchmarked head-to-head against scikit-learn's implementation to validate correctness and compare performance.

---

## Why This Project Exists

Most machine learning tutorials teach you to call `model.fit()` without explaining what happens inside. This project tears open the black box and implements every component by hand:

- Forward pass (prediction)
- Mean Squared Error loss function
- Gradient computation (calculus-derived, vectorized with NumPy)
- Gradient descent weight update loop
- Feature scaling (StandardScaler built from scratch)

---

## Project Structure

```
linear-regression-numpy/
│
├── linearregression.py       # Custom LinearRegression + StandardScaler classes
├── linear_regression.ipynb   # Notebook — EDA, training, evaluation, plots
├── Salary_dataset.csv        # Dataset (30 samples)
└── README.md
```

---

## Dataset

**Salary Dataset** — predicts annual salary from years of work experience.

| Feature         | Type  | Description              |
|-----------------|-------|--------------------------|
| YearsExperience | Float | Years of work experience |
| Salary          | Float | Annual salary (USD)      |

- 30 samples total — 24 training, 6 test
- 1 input feature, 1 continuous target

---

## How It Works

### StandardScaler (from scratch)
Standardizes features to zero mean and unit variance:

$$z = \frac{x - \mu}{\sigma}$$

```python
scaler  = StandardScaler()
x_train = scaler.fit_transform(x_train)  # learn mean/std, then scale
x_test  = scaler.transform(x_test)       # use same mean/std on test data
```

### LinearRegression (from scratch)

**Prediction:**
$$\hat{y} = X \cdot w + b$$

**Loss — Mean Squared Error:**
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

**Gradients:**
$$\frac{\partial MSE}{\partial w} = \frac{1}{n} X^T \cdot \text{error} \qquad \frac{\partial MSE}{\partial b} = \frac{1}{n} \sum \text{error}$$

**Weight Update:**
$$w = w - \alpha \cdot \frac{\partial MSE}{\partial w} \qquad b = b - \alpha \cdot \frac{\partial MSE}{\partial b}$$

---

## Results

| Model          | MSE     | R² Score |
|----------------|---------|----------|
| Custom (NumPy) | fill in | fill in  |
| Scikit-learn   | fill in | fill in  |

> Run the notebook to generate your actual numbers and fill these in.

---

## Visualizations

**Training Loss Curve** — MSE decreasing across iterations, confirming the model learns correctly.

**Custom vs Sklearn Regression Line** — both lines plotted against actual test data. They overlap almost perfectly, confirming the from-scratch implementation matches the industry standard.

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/linear-regression-numpy.git
cd linear-regression-numpy
```

**2. Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

**3. Launch the notebook**
```bash
jupyter notebook linear_regression.ipynb
```

---

## Dependencies

| Library      | Purpose                               |
|--------------|---------------------------------------|
| NumPy        | Matrix math and gradient computation  |
| Pandas       | Data loading and preprocessing        |
| Matplotlib   | Loss curve and prediction plots       |
| Seaborn      | Correlation heatmap                   |
| Scikit-learn | Benchmark comparison only             |

Scikit-learn is used **only** for benchmarking — the core model and scaler are pure NumPy.

---

## Key Concepts Demonstrated

- Gradient descent from first principles
- Vectorized NumPy operations — no Python loops for math
- Feature scaling — why it matters and how to build it without sklearn
- Inverse transforming predictions back to original scale
- Train/test split evaluation
- Comparing custom implementations to production libraries
- Loss history tracking and visualization

---

## What I Learned

Building this from scratch forced a deep understanding of:

- Why features must be scaled before gradient descent
- How the chain rule produces the gradient formulas
- Why `np.dot(X.T, error)` computes all weight gradients simultaneously
- The difference between mathematical gradient formulas and their simplified practical versions
- How `self.weights` and `self.bias` store the model's learned parameters

---

## Author

**kidnextdoor57**
Built as part of a machine learning fundamentals study project.

*If this helped you understand linear regression better, consider giving it a star ⭐*
