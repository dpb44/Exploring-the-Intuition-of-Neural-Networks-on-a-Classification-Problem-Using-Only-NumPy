# Exploring the Intuition of Neural Networks on a Classification Problem Using Only NumPy

## Overview
This project explores the intuition behind neural networks for multiclass classification using **only NumPy**, without high-level frameworks like TensorFlow or PyTorch. The goal is to classify the three **Iris species**—Setosa, Versicolor, and Virginica—based on petal and sepal measurements. We built a single-layer neural network using softmax activation, cross-entropy loss, and gradient descent to optimize model parameters.

![Image](https://github.com/user-attachments/assets/2fb3559c-e3c1-4e36-ba4c-fa77c3e3a221)

### Key Features:
- **Softmax activation** for multi-class classification.
- **Cross-entropy loss function** for model optimization.
- **Gradient Descent- Backpropagation** to update model parameters.
- **Vectorization and broadcasting** for computational efficiency.
- **Decision boundary visualization** to analyze model predictions.

## Dataset
The dataset consists of **150 samples**, each with **four numerical features**:
- **Sepal Length**
- **Sepal Width**
- **Petal Length**
- **Petal Width**

Each sample belongs to one of three classes:
- **Setosa (0)**
- **Versicolor (1)**
- **Virginica (2)**

These features are represented as $X$ in matrix form:

$$
X \in \mathbb{R}^{m \times n_x}
$$ 

where $m = 150$ (50 samples per species) and $n_x = 4$(features per sample).

```python
# Load the dataset using sklearn
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
```

## One-Hot Encoding
Since we are dealing with a multi-class classification problem, we convert categorical labels into **one-hot encoded vectors**.

```python
import numpy as np

m, K = y.shape[0], 3  # 3 classes

y_one_hot = np.zeros((m, K))
y_one_hot[np.arange(m), y] = 1
```
This transforms each label into a vector where only the corresponding class index is set to 1.

## Model Architecture
We use a **single-layer feed-forward neural network** with softmax activation.

### 1. Softmax Function
Since we have three distinct classes, we use the **softmax function** instead of the sigmoid function. The softmax function is given by:

$$
[g(\boldsymbol{t})]_k = \frac{e^{t_k}}{\sum_{j=1}^K e^{t_j}}
$$

where $\boldsymbol{t} = (t_k)_{k=1}^K$ represents the unnormalized class scores.

This function converts raw scores into probabilites.


```python
# Compute softmax activation
Z = np.dot(W.T, X) + b
numerator = np.exp(Z)
denominator = np.sum(numerator, axis=0, keepdims=True)
y_hat = (numerator / denominator).T
```

### 2. Cross-Entropy Loss Function
The loss function quantifies the difference between predicted and true labels:

$$
\mathcal{J}(\boldsymbol{W},\boldsymbol{b}) = -\frac{1}{m} \sum_{i=1}^m \sum_{j=1}^K \mathbf{y}_j^{(i)} \log(\widehat{y}^{(i)}_j)
$$


```python
# Compute loss
loss = np.sum(y_one_hot * np.log(y_hat), axis=1)
total_cost = - (1/m) * np.sum(loss)
```

### 3. Backpropagation Gradient Descent for Optimization
Using backpropagation, we compute gradients for weights and bias updates

$$
\nabla_{\boldsymbol{W}} \mathcal{J}(\boldsymbol{W},\boldsymbol{b}) = \frac{1}{m} (\widehat{\boldsymbol{Y}} - \mathbf{Y}) X^\top
$$

$$
\nabla_{\boldsymbol{b}} \mathcal{J}(\boldsymbol{W},\boldsymbol{b}) = \frac{1}{m} \sum_{i=1}^{m} (\widehat{\boldsymbol{y}}^{(i)} - \mathbf{y}^{(i)})
$$

We update parameters iteratively using:

$$
\boldsymbol{W} := \boldsymbol{W} - \alpha \nabla_{\boldsymbol{W}} \mathcal{J}(\boldsymbol{W},\boldsymbol{b})
$$

$$
\boldsymbol{b} := \boldsymbol{b} - \alpha \nabla_{\boldsymbol{b}} \mathcal{J}(\boldsymbol{W},\boldsymbol{b})
$$

![Image](https://github.com/user-attachments/assets/b30133d2-bc34-4c81-adf4-c40d6f2c35ea)

*Note: This image is only to show how the training updates the parameters though back propagation. It is not representative of the single-layer feed-forward neural network we have built.*

```python
# Compute gradients
W_grad = np.dot((y_hat - y_one_hot), X) / m
b_grad = np.sum((y_hat - y_one_hot), axis=1) / m
```

## Training the Model
We train the model using **gradient descent** over multiple iterations.

```python
# Training loop
costs = []
for i in range(iters):
    y_hat = p_model(X, W, b)
    cost = compute_cost(y, y_hat)
    W_grad, b_grad = compute_gradients(X, y, W, b)
    W -= lr * W_grad
    b -= lr * b_grad

    if i % 100 == 0:
        costs.append(cost)
        print(f"Cost after iteration {i}: {cost:.4f}")
```

## Model Evaluation & Results
We tested three feature sets:

| Feature Set        | Accuracy |
|------------------|----------|
| Petal Measurements | **96%**  |
| Sepal Measurements | **75%**  |
| Both Features      | **98%**  |

### Observations:
- **Petal measurements alone** perform better than **sepal measurements alone**.
- **Using both features gives the highest accuracy (98%)**.
- The **decision boundary** was influenced by the number of training iterations and learning rate.

## Decision Boundary Visualization
To visualize how the model classifies new data, we plot the decision boundary.

```python
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

y_pred = p_model(np.c_[xx.ravel(), yy.ravel()], W_trained, b_trained)
y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)

plt.contourf(xx, yy, y_pred, alpha=0.3, cmap=ListedColormap(['lightgreen', 'pink', 'coral']))
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("Decision Boundary")
plt.show()
```
  ![Image](https://github.com/user-attachments/assets/8cba0635-a9e4-481f-bc33-4d3db460e1ee)

  ![Image](https://github.com/user-attachments/assets/ae4dfc0b-54d1-41f1-9c39-c326301eeede)

### Decision Boundary Analysis:

- **Petal-only features**: Forms **well-defined decision regions** due to strong separability.
- **Sepal-only features**: The model perfoems poorly and is not able to form well-defined boundaires.

## Conclusion
- **Petal measurements provide a stronger predictive signal than sepal measurements**.
- **Gradient descent, softmax activation, and cross-entropy loss optimize the model effectively**.
- **Vectorization and broadcasting improve computational efficiency**.
- **Decision boundaries improve with more training iterations and proper hyperparameter tuning for the petal measurements**.


This project serves as a **minimal yet powerful demonstration** of how a neural network can be implemented from scratch, reinforcing mathematical intuition behind classification tasks.

---


## References

- [Iris Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- [Softmax Regression - Stanford CS229](https://cs229.stanford.edu/)
- [Medium Article by Srija Neogi - Exploring Multi-Class Classification using Deep Learning](https://medium.com/@srijaneogi31/exploring-multi-class-classification-using-deep-learning-cd3134290887)
- [Medium Article by LM Po - Backpropagation: The Backbone of Neural Network Training] (https://medium.com/@lmpo/backpropagation-the-backbone-of-neural-network-training-64946d6c3ae5)
---

## **Credits & Acknowledgments**  
This coursework was completed under the guidance of **Ms. Tatiana Buba** (Mathematics Professor).  

