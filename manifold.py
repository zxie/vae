import theano
import numpy as np
import cPickle as pickle
from vae import VAE
import matplotlib.pyplot as plt
from scipy.stats import norm

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='model file to load')
    parser.add_argument('dset', choices=['mnist'])
    args = parser.parse_args()

    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    if args.dset == 'mnist':
        S = (28, 28)
        M = 20

    manifold = np.zeros((S[0]*M, S[1]*M), dtype=theano.config.floatX)

    for z1 in xrange(M):
        for z2 in xrange(M):
            print z1, z2
            z = np.zeros((1, 2))
            # pass unit square through inverse Gaussian CDF
            z[0, 0] = norm.ppf(z1 * 1.0/M + 1.0/(M * 2))
            z[0, 1] = norm.ppf(z2 * 1.0/M + 1.0/(M * 2))
            z = np.array(z, dtype=theano.config.floatX)
            x_hat = model.decode(z)
            x_hat = x_hat.reshape(S)
            manifold[z1 * S[0]:(z1 + 1) * S[0],
                     z2 * S[1]:(z2 + 1) * S[1]] = x_hat

    plt.imshow(manifold, cmap='Greys_r')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
