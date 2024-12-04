import matplotlib.pyplot as plt


x = [0, 1, 2]
y = [1,1,1]
y2 = [1, 0.0164799, 0.000523583 ]


plt.scatter(x, y, color ='blue', label = 'TPR')
plt.plot(x, y, color='blue', linestyle='-')
plt.scatter(x, y2, color ='red', label = 'FPR')
plt.plot(x, y2, color='red', linestyle='-')



plt.xlabel('Training Stage')
plt.ylabel('Rate')
plt.title('The TPR and FPR at the different stages of training')
plt.legend()
plt.grid(False)
plt.xticks(x)

# Show the plot
plt.show()