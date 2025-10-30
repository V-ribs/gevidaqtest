import matplotlib.pyplot as plt
import numpy as np


def plot_npy_file(file_path):
    # Load the data from the .npy file
    data = np.load(file_path)

    # Check if the data is 1D or 2D
    if data.ndim == 1:
        plt.plot(data)
    # elif data.ndim == 2:
    # for i in range(data.shape[0]):
    # plt.plot(data[i], label=f'Channel {i+1}')
    # plt.legend()
    # else:
    # print("Unsupported data format")
    # return

    # Set plot labels and title
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("Plot of .npy File Data")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    # print("Usage: python open_patchclamp_file.py <path_to_npy_file>")
    # else:
    file_path = r"C:\Users\nccelie\Downloads\Ip2025-03-11_16-56-08.npy"
    plot_npy_file(file_path)
