import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Your code to get hprev, h, logits, and probs
# ...
def four_graphs(hpreact, hpreact_n, h):
# Convert the tensors to NumPy and flatten them for histogram plotting
    hpreact_numpy = hpreact.detach().numpy().flatten()
    hpreact_n_numpy = hpreact_n.detach().numpy().flatten()
    h_numpy = h.detach().numpy().flatten()

    # Create a figure with 2x2 grid of axes
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))

    # Plot histogram for hprev
    axs[0, 0].hist(hpreact_numpy, bins=50, color='blue', alpha=0.7)
    axs[0, 0].set_title('Histogram of hpreact values')
    axs[0, 0].set_xlabel('Value')
    axs[0, 0].set_ylabel('Frequency')

    axs[0, 1].hist(hpreact_n_numpy, bins=50, color='blue', alpha=0.7)
    axs[0, 1].set_title('Histogram of hpreact_n values')
    axs[0, 1].set_xlabel('Value')
    axs[0, 1].set_ylabel('Frequency')

    axs[1, 0].hist(h_numpy, bins=50, color='blue', alpha=0.7)
    axs[1, 0].set_title('Histogram of h values')
    axs[1, 0].set_xlabel('Value')
    axs[1, 0].set_ylabel('Frequency')

 

