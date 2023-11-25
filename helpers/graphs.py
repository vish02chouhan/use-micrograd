import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Your code to get hprev, h, logits, and probs
# ...
def four_graphs(data1,data2,data3, data4):
# Convert the tensors to NumPy and flatten them for histogram plotting
    data1_numpy = data1.detach().numpy().flatten()
    data2_numpy = data2.detach().numpy().flatten()
    data3_numpy = data3.detach().numpy().flatten()
    data4_numpy = data4.detach().numpy().flatten()

    # Create a figure with 2x2 grid of axes
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))

    # Plot histogram for hprev
    axs[0, 0].hist(data1_numpy, bins=50, color='blue', alpha=0.7)
    axs[0, 0].set_title('Histogram of data1 values')
    axs[0, 0].set_xlabel('Value')
    axs[0, 0].set_ylabel('Frequency')

    axs[0, 1].hist(data2_numpy, bins=50, color='blue', alpha=0.7)
    axs[0, 1].set_title('Histogram of data2 values')
    axs[0, 1].set_xlabel('Value')
    axs[0, 1].set_ylabel('Frequency')

    axs[1, 0].hist(data3_numpy, bins=50, color='blue', alpha=0.7)
    axs[1, 0].set_title('Histogram of data3 values')
    axs[1, 0].set_xlabel('Value')
    axs[1, 0].set_ylabel('Frequency')
    
    axs[1, 1].hist(data4_numpy, bins=50, color='blue', alpha=0.7)
    axs[1, 1].set_title('Histogram of data4 values')
    axs[1, 1].set_xlabel('Value')
    axs[1, 1].set_ylabel('Frequency')

def two_graphs(data1,data2):
# Convert the tensors to NumPy and flatten them for histogram plotting
    data1_numpy = data1.detach().numpy().flatten()
    data2_numpy = data2.detach().numpy().flatten()

    # Create a figure with 2x2 grid of axes
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Plot histogram for hprev
    axs[0].hist(data1_numpy, bins=50, color='blue', alpha=0.7)
    axs[0].set_title('Histogram of data1 values')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(data2_numpy, bins=50, color='blue', alpha=0.7)
    axs[1].set_title('Histogram of data2 values')
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('Frequency')
    
def one_graph(data):
    # Assuming 'data' is a tensor, convert it to NumPy and flatten it for histogram plotting
    data_numpy = data.detach().numpy().flatten()

    # Create a figure with a single axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot histogram for data
    ax.hist(data_numpy, bins=50, color='blue', alpha=0.7)
    ax.set_title('Histogram of data values')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')


 