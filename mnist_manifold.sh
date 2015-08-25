THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python vae.py --epochs 4000 --hdim 500 --outfile mnist_model.pk --dset mnist
THEANO_FLAGS=device=gpu,floatX=float32 python manifold.py mnist_model.pk mnist
