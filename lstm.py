import mxnet as mx
import mxnet.ndarray as nd

def init_lstm_args(xsize, hsize, ctx):
    return (
        nd.random_uniform(-.01, .01, shape=(xsize, hsize), ctx=ctx),
        nd.random_uniform(-.01, .01, shape=(xsize, hsize), ctx=ctx),
        nd.random_uniform(-.01, .01, shape=(xsize, hsize), ctx=ctx),
        nd.random_uniform(-.01, .01, shape=(hsize, hsize), ctx=ctx),
        nd.random_uniform(-.01, .01, shape=(hsize, hsize), ctx=ctx),
        nd.random_uniform(-.01, .01, shape=(hsize, hsize), ctx=ctx),
        nd.random_uniform(-.01, .01, shape=(xsize, hsize), ctx=ctx),
        nd.random_uniform(-.01, .01, shape=(hsize, hsize), ctx=ctx),
        nd.zeros(hsize, ctx=ctx),
        nd.zeros(hsize, ctx=ctx),
        nd.zeros(hsize, ctx=ctx),
        nd.zeros(hsize, ctx=ctx)
    )

def lstm(x, h, c, Wxi, Wxf, Wxo, Whi, Whf, Who, Wxc, Whc, bi, bf, bo, bc):
    i = nd.sigmoid(nd.dot(x, Wxi) + nd.dot(h, Whi) + bi)
    f = nd.sigmoid(nd.dot(x, Wxf) + nd.dot(h, Whf) + bf)
    o = nd.sigmoid(nd.dot(x, Wxo) + nd.dot(h, Who) + bo)
    c̃ = nd.tanh(nd.dot(x, Wxc) + nd.dot(h, Whc) + bc)
    c = f * c + i * c̃
    h = o * nd.tanh(c)
    return h, c

softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss()

class Model(object):
    def __init__(self, vsize, ctx='cpu'):
        self.vsize = vsize
        self.ctx = mx.gpu() if ctx == 'gpu' else mx.cpu()

        self.a1 = init_lstm_args(vsize, 1024, self.ctx)

        self.Wy = nd.random_uniform(-.01, .01, shape=(1024, vsize), ctx=self.ctx)
        self.by = nd.zeros(vsize, ctx=self.ctx)

        self.params = [*self.a1, self.Wy, self.by]

        for p in self.params:
            p.attach_grad()

    def forward(self, x):
        x = nd.one_hot(x, self.vsize)
        self.s1 = lstm(x, *self.s1, *self.a1)
        return nd.dot(self.s1[0], self.Wy) + self.by

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
        self.s1 = nd.zeros((batchsize, 1024), ctx=self.ctx), nd.zeros((batchsize, 1024), ctx=self.ctx)