import matplotlib.pyplot as plt

accuracies = [90.96, 67.18, 93.8, 55.3]
models = ['Random Forest', 'KNN', 'MLP', 'LR']
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of Different Models')
plt.ylim(0, 100)  # Set y-axis limits to 0-100%
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
