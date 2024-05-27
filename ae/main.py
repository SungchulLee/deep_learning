import torch.optim as optim

from global_name_space import ARGS
from load_data import load_data
from model import AE
from train import run_train_test_loop, load_model
from utils import show_ten_generated_images

def main():
    train_loader, test_loader = load_data()

    model = AE().to(ARGS.device)
    optimizer = optim.Adam(model.parameters(), lr=ARGS.lr)
    run_train_test_loop(model, train_loader, test_loader, optimizer)

    model = load_model(epoch=5)
    show_ten_generated_images(model)

if __name__ == "__main__":
    main()
