import mxnet as mx
import mxnet.ndarray as nd

def rnn(x, h, W, b):
    return nd.tanh(nd.dot(nd.concat(x, h, dim=1), W) + b)

softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss()

class Model(object):
    def __init__(self, vsize, ctx='cpu'):
        self.vsize = vsize
        self.ctx = mx.gpu() if ctx == 'gpu' else mx.cpu()

        self.W1 = nd.random_uniform(-.01, .01, shape=(vsize+1024, 1024), ctx=self.ctx)
        self.b1 = nd.zeros(1024, ctx=self.ctx)

        self.Wy = nd.random_uniform(-.01, .01, shape=(1024, vsize), ctx=self.ctx)
        self.by = nd.zeros(vsize, ctx=self.ctx)

        self.params = [self.W1, self.b1, self.Wy, self.by]

        for p in self.params:
            p.attach_grad()

    def forward(self, x):
        x = nd.one_hot(x, self.vsize)
        self.h1 = rnn(x, self.h1, self.W1, self.b1)
        return nd.dot(self.h1, self.Wy) + self.by

    def predict(self, x):
        return self.forward(nd.array([x], self.ctx, dtype=int))[0].asnumpy()

    def learn(self, xs):
        xs = [nd.array(x, self.ctx) for x in xs]
        with mx.autograd.record():
            loss = 0
            for i in range(len(xs)-1):
                p = self.forward(xs[i])
                if i >= 5:
                    loss = loss + softmax_cross_entropy(p, xs[i+1]) / (len(xs) - 5)
            loss = nd.mean(loss)

        loss.backward() # this overrides previous grad
        for p in self.params:
            m = p.grad.max().asscalar()
            p.grad[:] /= max(m, 1)
            p[:] = p - .1*p.grad

        return loss.asscalar()

    def reset(self, batchsize):
        self.h1 = nd.zeros((batchsize, 1024), ctx=self.ctx)