import matplotlib.pyplot as plt

def draw(X, Y, Z, x0_trace, y0_trace, contour=True, x_min=0, x_max=5, y_min=0, y_max=5):
    _, ax = plt.subplots(figsize=(15,6))
    ax.plot(x0_trace,y0_trace,'--or')
    if contour:
        contours = ax.contour(X, Y, Z, 3, colors='black')
        plt.clabel(contours, inline=True, fontsize=20)
    else:
        ax.imshow(Z, origin='lower', extent=(x_min,x_max,y_min,y_max), cmap='coolwarm')
        ax.axis('off')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()