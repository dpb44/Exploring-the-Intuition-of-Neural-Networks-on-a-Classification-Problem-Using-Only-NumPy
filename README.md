# Exploring the Intuition of Neural Networks on a Classification Problem Using Only NumPy

## Overview

This project explores the **intuition behind neural networks** for **multiclass classification** using **only NumPy**—without high-level frameworks like TensorFlow or PyTorch. The objective is to classify the three Iris species (**Setosa, Versicolor, and Virginica**) using **petal and sepal measurements**.

We apply fundamental mathematical concepts such as **softmax activation, cross-entropy loss, and gradient descent updates** to train the model. Additionally, **vectorization and broadcasting** are utilized to improve computational efficiency by avoiding explicit loops.

No feature normalization was performed, allowing us to analyze the raw impact of feature magnitudes on classification performance.

![Image](https://github.com/user-attachments/assets/811fd4b9-30a6-4ea3-b2cf-5f61c17caf54)

---

## Mathematical Foundations


### **1. Softmax Function**
For a given input vector $z$, the softmax function outputs probabilities for each class:

$$
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

where $z_i$ represents the score for class $i$.

### **2. Cross-Entropy Loss**
The loss function measures how well the predicted probabilities align with the true labels:

$$
L = -\sum_{i} y_i \log(\hat{y}_i)
$$

where $y_i$ is the actual class label (one-hot encoded) and $\hat{y}_i$ is the predicted probability for class $i$.

### **3. Gradient Descent Updates**
To minimize the loss function, we compute gradients and update parameters:

$$
W = W - \eta \frac{\partial L}{\partial W}, \quad b = b - \eta \frac{\partial L}{\partial b}
$$

where $\eta$ is the learning rate.

---

## Dataset

The dataset consists of **150 samples**, with four numerical features:

- **Sepal Length**
- **Sepal Width**
- **Petal Length**
- **Petal Width**

Each sample belongs to one of **three classes**:

- **Setosa (0)**
- **Versicolor (1)**
- **Virginica (2)**

---

## Methodology

### **1. Neural Network Implementation (Without High-Level Libraries)**

- Used a **single-layer neural network** with a **softmax activation** for multiclass classification.
- Applied **cross-entropy loss** as the cost function.
- Implemented **gradient descent** for optimization.
- Utilized **vectorization and broadcasting** for efficient computations.
- No feature normalization was applied aince all measurements were in centimeters.

### **2. Feature Importance Analysis**

- Trained separate models using **only sepal measurements** and **only petal measurements**, and one using both.
- Compared decision boundary formations and classification accuracy.

### **3. Decision Boundary Visualization**

- Plotted **decision boundaries** using **scatter plots** of sepal and petal features.
- Observed the effect of different learning rates and iterations on boundary formation.
- Notably, the **petal measurements** led to **sharper and more distinct class separation**, whereas the **sepal measurements resulted in overlapping boundaries**.

---

## Results

### **1. Accuracy Comparison:**

| Feature Set        | Accuracy |
| ------------------ | -------- |
| Petal Measurements | **96%**  |
| Sepal Measurements | **75%**  |
| Both Features      | **98%**  |

**Key Observations:**
- The model achieves **highest accuracy (98%)** when using both features.
- **Petal measurements alone** performed significantly better than **sepal measurements alone**.
- The decision boundary was significantly influenced by the number of training iterations and learning rate.

### **2. Decision Boundary Analysis:**

- **Petal-only features**: Forms **well-defined decision regions** due to strong separability.
- **Sepal-only features**: Results in **overlapping boundaries**, making classification harder.
- **Both features combined**: Minimizes the **cost function more effectively** but doesn't significantly alter decision boundaries.

  ![Image](https://github.com/user-attachments/assets/8cba0635-a9e4-481f-bc33-4d3db460e1ee)

![Image](https://github.com/user-attachments/assets/ae4dfc0b-54d1-41f1-9c39-c326301eeede)

---

## Code Implementation

### **1. Train the Model**

```python
# Train Neural Network Model (Single-Layer)
W_trained, b_trained, costs = train(X, y, lr=0.2, iters=20000, W=np.random.rand(4, 3), b=np.random.rand(3,))
```

### **2. Plot Decision Boundaries**

```python
# Generate decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

y_pred = p_model(np.c_[xx.ravel(), yy.ravel()], W_trained, b_trained)
y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)

# Plot
printf.contourf(xx, yy, y_pred, alpha=0.3, cmap=ListedColormap(['lightgreen', 'pink', 'coral']))
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("Decision Boundary")
plt.show()
```

---

## Conclusion

- **Petal measurements** provide a **stronger predictive signal** than sepal measurements.
- **Gradient descent, softmax activation, and cross-entropy loss effectively optimized the model**.
- **Vectorization and broadcasting significantly improve computational efficiency**.
- **Decision boundaries improve with increased iterations and learning rate tuning**.

---

## References

- [Iris Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- [Softmax Regression - Stanford CS229](https://cs229.stanford.edu/)
- [Medium Article by Srija Neogi - Exploring Multi-Class Classification using Deep Learning](https://medium.com/@srijaneogi31/exploring-multi-class-classification-using-deep-learning-cd3134290887)



