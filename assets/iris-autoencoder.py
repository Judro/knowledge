#! /usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 2),
            torch.nn.ReLU(), # <-- Funktion variieren
            )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.ReLU(), # <-- Funktion variieren
            )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    
def main():
    # === Daten laden ===
    X, y = load_iris(return_X_y=True)
    #X = PCA(4).fit_transform(X) # <-- ein/aus kommentieren
    train = TensorDataset(torch.as_tensor(X, dtype=torch.float32))
    train_loader = DataLoader(train,
                              batch_size=16,  # <-- Zahl variieren
                              shuffle=True)

    
    # === Modell trainieren ===    
    model = AutoEncoder()
    loss_fun = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr = 0.004)  # <-- Zahl variieren
    print("initial encoder weights\n", model.encoder[0].weight.detach().numpy())

    epochs=20 # <-- Zahl variieren
    losses=[]
    for epoch in range(epochs):
        for sample in train_loader:
            reconstructed = model(sample[0])
            loss = loss_fun(reconstructed, sample[0])
            # recompute gradients:
            optimizer.zero_grad() # clean gradient cache
            loss.backward() # do backprop
            optimizer.step() # use gradient to learn
            losses.append(loss.detach().numpy())


    # === Ergebnisse angucken ===
    print("found encoder weights\n", model.encoder[0].weight.detach().numpy())
            
    # geht das in den folgenden zwei Zeilen nicht auch schöner?
    reconstruction = np.array([model(torch.tensor(x, dtype=torch.float32)).detach().numpy() for x in X])
    latents = np.array([model.encoder(torch.tensor(x, dtype=torch.float32)).detach().numpy() for x in X])

    # plot in X-PCA coordinates:
    pca = PCA(2).fit(X)
    X = pca.transform(X)
    reconstruction_ = pca.transform(reconstruction)
    reconstruction = PCA(2).fit_transform(reconstruction)
    
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
    ax1.scatter(X[:, 0], X[:, 1], c=y)
    ax1.set_title("Original data, first two PCs")
    ax2.scatter(reconstruction_[:, 0], reconstruction_[:, 1], c=y)
    ax2.set_title("Reconstruction, same projection ")
    ax3.scatter(reconstruction[:, 0], reconstruction[:, 1], c=y)
    ax3.set_title("Reconstruction, first two PCs")
    ax4.scatter(latents[:, 0], latents[:, 1], c=y) #XXX
#    ax4.scatter(latents[:, 0], np.zeros_like(latents[:,0]), c=y)
    ax4.set_title("Latent vectors")
    ax5.plot(losses)
    ax5.set_title("MSE Loss")
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    
if(__name__ == "__main__"):
    main()
