import numpy as np
import matplotlib.pyplot as plt

from global_name_space import ARGS

def draw(theta_trace, loss_trace):
    _, (ax0, ax1) = plt.subplots(1,2,figsize=(12,3))

    theta_fig = np.linspace(-1.5, 1.5)
    loss_fig = ARGS.compute_loss(theta_fig)
    ax0.plot(theta_fig,loss_fig)
    ax0.plot(theta_trace,loss_trace,'--*r',label='parameter')

    ax1.plot(loss_trace,'-*',label="loss")

    for ax in (ax0, ax1):
        ax.legend()
    plt.tight_layout()
    plt.show()