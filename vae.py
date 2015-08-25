import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle
from mlp import GaussianMLP, BernoulliMLP
from utils import kld_unit_mvn, load_dataset, floatX

ADAGRAD_EPS = 1e-10  # for stability

class VAE(object):

    def __init__(self, xdim, args, dec='bernoulli'):
        self.xdim = xdim
        self.hdim = args.hdim
        self.zdim = args.zdim
        self.lmbda = args.lmbda  # weight decay coefficient * 2
        self.x = T.matrix('x', dtype=floatX)
        self.eps = T.matrix('eps', dtype=floatX)

        # XXX make this more general
        self.enc_mlp = GaussianMLP(self.x, self.xdim, self.hdim, self.zdim, nlayers=args.nlayers, eps=self.eps)
        if dec == 'bernoulli':
            # log p(x | z) defined as -CE(x, y) = dec_mlp.cost(y)
            self.dec_mlp = BernoulliMLP(self.enc_mlp.out, self.zdim, self.hdim, self.xdim, nlayers=args.nlayers, y=self.x)
        elif dec == 'gaussian':
            self.dec_mlp = GaussianMLP(self.enc_mlp.out, self.zdim, self.hdim, self.xdim, nlayers=args.nlayers, y=self.x)
        else:
            raise RuntimeError('unrecognized decoder %' % dec)

        self.cost = (-T.sum(kld_unit_mvn(self.enc_mlp.mu, self.enc_mlp.var)) + self.dec_mlp.cost) / args.batch_size
        self.params = self.enc_mlp.params + self.dec_mlp.params
        print(self.params)
        self.gparams = [T.grad(self.cost, p) + self.lmbda * p for p in self.params]
        self.gaccums = [theano.shared(value=np.zeros(p.get_value().shape, dtype=floatX)) for p in self.params]

        # XXX using adagrad update as described in paper, could try other optimizers
        self.updates = [
                (param, param - args.lr * gparam / T.sqrt(gaccum + T.square(gparam) + ADAGRAD_EPS))
                for param, gparam, gaccum in zip(self.params, self.gparams, self.gaccums)
        ]
        self.updates += [
            (gaccum, gaccum + T.square(gparam))
            for gaccum, gparam in zip(self.gaccums, self.gparams)
        ]

        self.train = theano.function(
            inputs=[self.x, self.eps],
            outputs=self.cost,
            updates=self.updates
        )
        self.test = theano.function(
            inputs=[self.x, self.eps],
            outputs=self.cost,
            updates=None
        )
        # can be used for semi-supervised learning for example
        self.encode = theano.function(
            inputs=[self.x, self.eps],
            outputs=self.enc_mlp.out
        )
        # use this to sample
        self.decode = theano.function(
            inputs=[self.enc_mlp.out],
            outputs=self.dec_mlp.out
        )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100)
    # XXX using sample size of one
    parser.add_argument('--nlayers', default=1, type=int, help='number of hidden layers in MLP before output layers')
    parser.add_argument('--hdim', default=500, type=int, help='dimension of hidden layer')
    parser.add_argument('--zdim', default=2, type=int, help='dimension of continuous latent variable')
    parser.add_argument('--lmbda', default=0.001, type=float, help='weight decay coefficient')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', default=1000, type=int, help='number of passes over dataset')
    parser.add_argument('--print_every', default=100, type=int, help='how often to print cost')
    parser.add_argument('--save_every', default=1, type=int, help='how often to save model (in terms of epochs)')
    parser.add_argument('--outfile', default='vae_model.pk', help='output file to save model to')
    parser.add_argument('--dset', default='mnist', choices=['mnist'],
            help='dataset to use')
    args = parser.parse_args()
    print(args)

    # run SGVB algorithm

    # N x d
    data = load_dataset(dset=args.dset)
    train_x, train_y = data['train']
    #print(train_x[0, :])  # values in [0, 1]
    #print(train_y[0:10])  # seems to already be shuffled
    valid_x, valid_y = data['valid']

    decs = {'mnist': 'bernoulli'}
    model = VAE(train_x.shape[1], args, dec=decs[args.dset])

    expcost = None
    num_train_batches = train_x.shape[0] / args.batch_size
    num_valid_batches = valid_x.shape[0] / args.batch_size
    valid_freq = num_train_batches

    for b in xrange(args.epochs * num_train_batches):
        k = b % num_train_batches
        x = train_x[k * args.batch_size:(k + 1) * args.batch_size, :]
        eps = np.random.randn(x.shape[0], args.zdim).astype(floatX)
        cost = model.train(x, eps)
        if not expcost:
            expcost = cost
        else:
            expcost = 0.01 * cost + 0.99 * expcost
        if (b + 1) % args.print_every == 0:
            print('iter %d, cost %f, expcost %f' % (b + 1, cost, expcost))
        if (b + 1) % valid_freq == 0:
            valid_cost = 0
            for l in xrange(num_valid_batches):
                x_val = valid_x[l * args.batch_size:(l + 1) * args.batch_size, :]
                eps_val = np.zeros((x_val.shape[0], args.zdim), dtype=floatX)
                valid_cost = valid_cost + model.test(x_val, eps_val)
            valid_cost = valid_cost / num_valid_batches
            print('valid cost: %f' % valid_cost)
        if (b + 1) % (num_train_batches * args.save_every) == 0:
            print('saving model')
            with open(args.outfile, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    # XXX just pickling the entire model for now
    print('saving final model')
    with open(args.outfile, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
