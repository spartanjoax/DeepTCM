
import os
import numpy as np
from keras.callbacks import EarlyStopping
from data.datasets import NASA_Dataset
from models.generative import ConditionalVAE, train_generative_model
from joblib import dump
import matplotlib.pyplot as plt

def train_cvae(sliding_window_size=250, sliding_window_stride=25, output_dir='vae_results', epochs=200):
    print("Loading Dataset...")
    dataset = NASA_Dataset(split='train', signal_group='all', 
                           sliding_window_size=sliding_window_size, 
                           sliding_window_stride=sliding_window_stride)
    
    train_generative_model(ConditionalVAE, dataset, sliding_window_size, sliding_window_stride, 
                           latent_dim=16, epochs=epochs, output_dir=output_dir)

def main():
    train_cvae()

if __name__ == "__main__":
    main()
