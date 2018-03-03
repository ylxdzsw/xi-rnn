import mxnet as mx
import mxnet.ndarray as nd

def init_gru_args(xsize, hsize, ctx):
    return (
        nd.random_uniform(-.01, .01, shape=(xsize, hsize), ctx=ctx),
        nd.random_uniform(-.01, .01, shape=(xsize, hsize), ctx=ctx),
        nd.random_uniform(-.01, .01, shape=(hsize, hsize), ctx=ctx),
        nd.random_uniform(-.01, .01, shape=(hsize, hsize), ctx=ctx),
        nd.random_uniform(-.01, .01, shape=(xsize, hsize), ctx=ctx),
        nd.random_uniform(-.01, .01, shape=(hsize, hsize), ctx=ctx),
        nd.zeros(hsize, ctx=ctx),
        nd.zeros(hsize, ctx=ctx),
        nd.zeros(hsize, ctx=ctx)
    )

def gru(x, h, Wxr, Wxz, Whr, Whz, Wxh, Whh, br, bz, bh):
    r = nd.sigmoid(nd.dot(x, Wxr) + nd.dot(h, Whr) + br)
    z = nd.sigmoid(nd.dot(x, Wxz) + nd.dot(h, Whz) + bz)
    h̃ = nd.tanh(nd.dot(x, Wxh) + r * nd.dot(h, Whh) + bh)
    return z * h + (1 - z) * h̃

softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss()

class Model(object):
    def __init__(self, vsize, ctx='cpu'):
        self.vsize = vsize
        self.ctx = mx.gpu() if ctx == 'gpu' else mx.cpu()

        self.a1 = init_gru_args(vsize, 1024, self.ctx)

        self.Wy = nd.random_uniform(-.01, .01, shape=(1024, vsize), ctx=self.ctx)
        self.by = nd.zeros(vsize, ctx=self.ctx)

        self.params = [*self.a1, self.Wy, self.by]

        for p in self.params:
            p.attach_grad()

    def forward(self, x):
        x = nd.one_hot(x, self.vsize)
        self.h1 = gru(x, self.h1, *self.a1)
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
            p[:] = p - .01*p.grad

        return loss.asscalar()

    def reset(self, batchsize):
        self.h1 = nd.zeros((batchsize, 1024), ctx=self.ctx)