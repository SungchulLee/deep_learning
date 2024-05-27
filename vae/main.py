import torch.optim as optim

from global_name_space import ARGS
from load_data import load_data
from model import VAE
from train import run_train_test_loop, load_model
from utils import show_ten_generated_images

# def save_model(model, epoch):
#     torch.save(model.state_dict(), f"{ARGS.path}/model_weights_{epoch}.pth")

# def load_model(epoch):
#     model = VAE().to(ARGS.device)
#     model.load_state_dict(torch.load(f"{ARGS.path}/model_weights_{epoch}.pth"))
#     return model

# # Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#     return BCE + KLD

# def train(epoch, model, train_loader, optimizer):
#     # This sets the model to training mode.
#     model.train()  # Sets the model to training mode
#     train_loss = 0

#     for batch_idx, (data, _) in enumerate(train_loader):
#         data = data.to(ARGS.device)  # Move data to the specified device (GPU or CPU)
#         optimizer.zero_grad()  # Zero out the gradients from the previous iteration
#         recon_batch, mu, logvar = model(data)  # Forward pass through the VAE
#         loss = loss_function(recon_batch, data, mu, logvar)  # Calculate the loss
#         loss.backward()  # Backward pass to compute gradients
#         train_loss += loss.item()  # Accumulate the loss value
#         optimizer.step()  # Update the model parameters using the optimizer

#         if batch_idx % ARGS.log_interval == 0:
#             # Print training progress every specified number of batches
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader),
#                 loss.item() / len(data)))

#     # Print the average loss for the entire epoch
#     print('====> Epoch: {} Average loss: {:.4f}'.format(
#           epoch, train_loss / len(train_loader.dataset)))
#     save_model(model,epoch)
    
# def test(epoch, model, test_loader):
#     #  This sets the model to evaluation mode.
#     model.eval()  # Sets the model to evaluation mode
#     test_loss = 0

#     with torch.no_grad():
#         for i, (data, _) in enumerate(test_loader):
#             data = data.to(ARGS.device)  # Move data to the specified device (GPU or CPU)
#             recon_batch, mu, logvar = model(data)  # Forward pass through the VAE
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()

#             if i == 0:
#                 # Visualize the reconstruction of the first batch
#                 n = min(data.size(0), 8)
#                 comparison = torch.cat([data[:n],
#                                         recon_batch.view(ARGS.batch_size, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(),
#                             f'{ARGS.path_reconstructed_images}/reconstruction_' + str(epoch) + '.png', nrow=n)

#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))

# def run_train_test_loop(model, train_loader, test_loader, optimizer):
#     for epoch in range(1, ARGS.epochs + 1):
#         train(epoch, model, train_loader, optimizer)
#         test(epoch, model, test_loader)

#         # This context manager ensures that gradients are not computed
#         # during the generation of samples, saving memory and computation.
#         with torch.no_grad():
#             # Generate and save samples from the latent space
#             sample = torch.randn(64, 20).to(ARGS.device)

#             # This decodes the random samples using the VAE's decoder.
#             # The .cpu() moves the generated samples back to the CPU
#             # for visualization.
#             sample = model.decode(sample).cpu()

#             # This reshapes and saves the generated samples as images.
#             # The view function reshapes the tensor to have
#             # dimensions (64, 1, 28, 28) suitable for image display,
#             # and save_image saves the images to the specified file path.
#             # The file names include the epoch number for differentiation.
#             save_image(sample.view(64, 1, 28, 28),
#                        f'{ARGS.path_generated_images}/sample_' + str(epoch) + '.png')

def main():
    train_loader, test_loader = load_data()

    model = VAE().to(ARGS.device)
    optimizer = optim.Adam(model.parameters(), lr=ARGS.lr)
    run_train_test_loop(model, train_loader, test_loader, optimizer)

    model = load_model(epoch=100)

    show_ten_generated_images(model)

if __name__ == "__main__":
    main()
