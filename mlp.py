import numpy as np
import theano
import theano.tensor as T
from utils import log_diag_mvn

# XXX
rng = np.random.RandomState(1234)

class HiddenLayer(object):

    # adapted from http://deeplearning.net/tutorial/mlp.html

    def __init__(self, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, prefix=''):
        self.n_in = n_in
        self.n_out = n_out

        if W is None:
            # NOTE tried glorot init and randn and glorot init worked better
            # after 1 epoch with adagrad
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                #rng.randn(n_in, n_out) * 0.01,
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name=prefix+'_W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=prefix+'_b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class _MLP(object):

    # building block for MLP instantiations defined below

    def __init__(self, x, n_in, n_hid, nlayers=1, prefix=''):
        self.nlayers = nlayers
        self.hidden_layers = list()
        inp = x
        for k in xrange(self.nlayers):
            hlayer = HiddenLayer(
                input=inp,
                n_in=n_in,
                n_out=n_hid,
                activation=T.tanh,
                prefix=prefix + ('_%d' % (k + 1))
            )
            n_in = n_hid
            inp = hlayer.output
            self.hidden_layers.append(hlayer)

        self.params = [param for l in self.hidden_layers for param in l.params]
        self.input = input
        # NOTE output layer computed by instantations


class GaussianMLP(_MLP):

    def __init__(self, x, n_in, n_hid, n_out, nlayers=1, y=None, eps=None):
        super(GaussianMLP, self).__init__(x, n_in, n_hid, nlayers=nlayers, prefix='GaussianMLP_hidden')
        self.mu_layer = HiddenLayer(
            input=self.hidden_layers[-1].output,
            n_in=self.hidden_layers[-1].n_out,
            n_out=n_out,
            activation=None,
            prefix='GaussianMLP_mu'
        )
        # log(sigma^2)
        self.logvar_layer = HiddenLayer(
            input=self.hidden_layers[-1].output,
            n_in=self.hidden_layers[-1].n_out,
            n_out=n_out,
            activation=None,
            prefix='GaussianMLP_logvar'
        )
        self.mu = self.mu_layer.output
        self.var = T.exp(self.logvar_layer.output)
        self.sigma = T.sqrt(self.var)
        self.params = self.params + self.mu_layer.params +\
            self.logvar_layer.params
        # for use as encoder
        if eps:
            assert(y is None)
            # XXX separate reparametrization
            self.out = self.mu + self.sigma * eps
        # for use as decoder
        if y:
            assert(eps is None)
            # XXX specific to [0, 1] outputs
            self.out = T.nnet.sigmoid(self.mu)
            self.cost = -T.sum(log_diag_mvn(self.out, self.var)(y))

class BernoulliMLP(_MLP):

    def __init__(self, x, n_in, n_hid, n_out, nlayers=1, y=None):
        super(BernoulliMLP, self).__init__(x, n_in, n_hid, nlayers=nlayers, prefix='BernoulliMLP_hidden')
        self.out_layer = HiddenLayer(
            input=self.hidden_layers[-1].output,
            n_in=self.hidden_layers[-1].n_out,
            n_out=n_out,
            activation=T.nnet.sigmoid,
            prefix='BernoulliMLP_y_hat'
        )
        self.params = self.params + self.out_layer.params
        if y:
            self.out = self.out_layer.output
            self.cost = T.sum(T.nnet.binary_crossentropy(self.out, y))
