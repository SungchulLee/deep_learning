from global_name_space import ARGS
from train import train
from utils import draw

def main():
    theta = 1.2
    theta, theta_trace, loss_trace = train(theta)
    draw(theta_trace, loss_trace)

if __name__ == "__main__":
    main()