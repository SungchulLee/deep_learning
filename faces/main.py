from global_name_space import *
from load_data import load_data
from utils import show_one_image_from_datafolder, show_four_images_from_dataloader

def main():
    dataloader, df_landmarks  = load_data()

    show_one_image_from_datafolder(df_landmarks, n=65)
    show_four_images_from_dataloader(dataloader)

if __name__ == "__main__":
    main()