import time, random
from nn import MLP

random.seed(0)
for nin in [64, 196, 784]:
    m = MLP(nin, [16, 10])
    x = [random.random() for _ in range(nin)]
    t = time.time()
    for _ in range(5):
        out = m(x)
        loss = sum(o ** 2 for o in out)
        for p in m.parameters():
            p.grad = 0.0
        loss.backward()
    print(nin, f"{(time.time() - t) / 5 * 1000:.1f} ms per sample")