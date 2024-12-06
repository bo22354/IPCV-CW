import matplotlib.pyplot as plt
import numpy as np

# Data for the bar graph
categories = ['1-5', '6-10', '11-15', '16-20', '21-25', '26-30']
values1 = [np.nan, 0.11, 0.18, 0.57, 1.1, np.nan] #centres error
values2 = [np.nan, 0.12, 0.23, 0.38, 0.51,  np.nan]  #raddi error

x = np.arange(len(categories))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
bar1 = ax.bar(x - width/2, values1, width, label='Centres', color='skyblue', edgecolor='black')
bar2 = ax.bar(x + width/2, values2, width, label='Radii', color='salmon', edgecolor='black')



# Add some labels and title
ax.set_xlabel('Radius')
ax.set_ylabel('Root-Mean Square Error')
ax.set_title('Effects Different Radius has on Esitmation Errors')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Display the graph
plt.show()























categories = ['1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7,8']
values1 = [0.15, 0.15, 0.10, 0.14, 0.14, 0.17, 0.13 ] #centres error
values2 = [0.22, 0.33, 0.28, 0.27, 0.25, 0.24, 0.22]  #raddi error


# Parameters for the bars
x = np.arange(len(categories))  # X locations for the groups
bar_width = 0.35  # Width of each bar

# Creating the grouped bar chart
plt.figure(figsize=(8, 5))
plt.bar(x - bar_width/2, values1, bar_width, label='Centres', color='skyblue', edgecolor='black')
plt.bar(x + bar_width/2, values2, bar_width, label='Radii', color='salmon', edgecolor='black')

# Adding labels and title
plt.xlabel('Seperation', fontsize=12)
plt.ylabel('Root-Mean Square Error', fontsize=12)
plt.title('Effects Different Seperation has on Esitmation Errors', fontsize=14)
plt.xticks(x, categories)  # Set category names at the x positions
plt.legend()  # Add a legend to differentiate the bars

# Displaying the values above the bars
for i, (v1, v2) in enumerate(zip(values1, values2)):
    plt.text(i - bar_width/2, v1 + 0.5, str(v1), ha='center', fontsize=10)
    plt.text(i + bar_width/2, v2 + 0.5, str(v2), ha='center', fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()