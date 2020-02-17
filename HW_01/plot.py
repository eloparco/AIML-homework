import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_data_and_boundaries(X_train, y_train, classifier, ax, x_label, y_label, title):
    # set bound as a % of average of ranges for x and y
    bound = 0.1 * np.average(np.ptp(X_train, axis=0))
    # set step size as a % of the average of ranges for x and y 
    step = 0.005 * np.average(np.ptp(X_train, axis=0))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max]
    x_min, x_max = X_train[:, 0].min() - bound, X_train[:, 0].max() + bound
    y_min, y_max = X_train[:, 1].min() - bound, X_train[:, 1].max() + bound
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold,
                edgecolor='k', s=20)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
def print_scatter_legend(fig, class_names, colors):
    handles = []
    for i in range(len(class_names)):
        legend_entry, = plt.plot([], marker="o", color=colors[i], linewidth=0, label=class_names[i])
        handles.append(legend_entry)
    fig.legend(loc='lower right')
