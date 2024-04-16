from load_data import load_data
from train import train
from utils import draw

def main():
    X, Y, Z = load_data()
    x0_trace, y0_trace = train(x0=2, y0=4)
    draw(X, Y, Z, x0_trace, y0_trace)

if __name__ == "__main__":
    main()