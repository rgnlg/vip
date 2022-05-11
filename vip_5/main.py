import matplotlib.pyplot as plt
from solution import Assignment


def main():
    solution = Assignment('./images/')

    solution.k_means()
    solution.otsu()
    solution.denoising()
    solution.denoising_passes()
    solution.k_means_with_larger_k()
    solution.chan_vese()

    plt.show()


if __name__ == '__main__':
    main()
