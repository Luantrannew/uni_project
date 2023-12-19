import numpy as np
import matplotlib.pyplot as plt

# Tạo dữ liệu mẫu cho hai lớp
np.random.seed(42)

# Cụm dữ liệu 1
data1 = np.random.randn(10, 2) + np.array([2, 2])

# Cụm dữ liệu 2
data2 = np.random.randn(10, 2) + np.array([6, 6])

# Gộp dữ liệu
X = np.concatenate((data1, data2), axis=0)

# Gán nhãn cho dữ liệu
y = np.array([0] * 10 + [1] * 10)

# Thêm cột bias vào dữ liệu
X_bias = np.c_[np.ones((X.shape[0], 1)), X]

# Khởi tạo trọng số ngẫu nhiên
np.random.seed(seed=42)
weights = np.random.rand(X_bias.shape[1])

# Hàm kích hoạt (step function)
def activation_function(z):
    return 1 if z >= 0 else 0

# Hàm dự đoán
def predict(sample, weights):
    z = np.dot(sample, weights)
    return activation_function(z)

# Huấn luyện perceptron
epochs = 100
learning_rate = 0.1

for epoch in range(epochs):
    for i in range(X_bias.shape[0]):
        prediction = predict(X_bias[i], weights)
        error = y[i] - prediction
        weights += learning_rate * error * X_bias[i]

# Đồ thị hóa kết quả
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Vẽ đường phân lớp tuyến tính
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = np.array([predict(np.array([1, xi, yi]), weights) for xi, yi in zip(xx.ravel(), yy.ravel())])

# Đồ thị hóa đường phân lớp
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', linewidths=3)
plt.title('Perceptron Classifier')
plt.show()
