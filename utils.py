import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle

'''
helper functions
'''

floatX = theano.config.floatX

# dataset

# XXX dataset parameters
MNIST_PATH = 'data/mnist.pkl.gz'

def load_dataset(dset='mnist'):
    if dset == 'mnist':
        import gzip
        f = gzip.open(MNIST_PATH, 'rb')
        train_set, valid_set, test_set = pickle.load(f)
        f.close()
        data = {'train': train_set, 'valid': valid_set, 'test': test_set}
    else:
        raise RuntimeError('unrecognized dataset: %s' % dset)
    return data

# costs

def kld_unit_mvn(mu, var):
    # KL divergence from N(0, I)
    return (mu.shape[1] + T.sum(T.log(var), axis=1) - T.sum(T.square(mu), axis=1) - T.sum(var, axis=1)) / 2.0

def log_diag_mvn(mu, var):
    def f(x):
        # expects batches
        k = mu.shape[1]
        logp = (-k / 2.0) * np.log(2 * np.pi) - 0.5 * T.sum(T.log(var), axis=1) - T.sum(0.5 * (1.0 / var) * (x - mu) * (x - mu), axis=1)
        return logp
    return f

# test things out

if __name__ == '__main__':
    f = log_diag_mvn(np.zeros(2), np.ones(2))
    x = T.vector('x')
    g = theano.function([x], f(x))
    print g(np.zeros(2))
    print g(np.random.randn(2))

    mu = T.vector('mu')
    var = T.vector('var')
    j = kld_unit_mvn(mu, var)
    g = theano.function([mu, var], j)
    print g(np.random.randn(2), np.abs(np.random.randn(2)))
