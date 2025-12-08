# 📉 Linear Regression From Scratch

A pedagogical implementation of Simple Linear Regression using pure Python and NumPy. This project demonstrates the core concepts of Gradient Descent and Mean Squared Error (MSE) minimization without relying on high-level libraries like scikit-learn for the core logic.

## 🚀 Features

- **Pure NumPy Implementation**: Understand the math behind the magic.
- **Gradient Descent**: Custom implementation of the optimization algorithm.
- **Visualization**: Matplotlib plots to visualize the regression line and data points.
- **Interactive Notebook**: Step-by-step derivation and code.

## 📊 Performance Metrics

Since this is a "from scratch" implementation, we evaluate it against synthetic data to ensure correctness.

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Mean Squared Error (MSE)** | **~300-400** | Dependent on noise level in synthetic data |
| **Convergence** | **1500 Iterations** | Reaches optimal weights with Learning Rate = 0.01 |
| **Implementation** | **Vectorized** | Uses `np.dot` for efficient computation |

## 🏗️ Project Structure

```
LINEAR_REGRESSION_FROM_SCRATCH/
└── main.ipynb          # Jupyter Notebook containing the implementation and visualization
```

## 🛠️ Technologies Used

- **Python**
- **NumPy** (Matrix operations)
- **Matplotlib** (Visualization)
- **Pandas** (Data handling)

## 🚀 Usage

Open the `main.ipynb` file in Jupyter Notebook or VS Code to run the simulation.

```bash
jupyter notebook main.ipynb
```

## 👤 Author
**Pawan Parida**
