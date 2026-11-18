import random
import math
import time
from sklearn.datasets import load_digits
from micrograd.littlegrad.nn import MLP

random.seed(0)

start = time.perf_counter()

def accuracy(model, samples):
    correct = 0
    for x, y in samples:
        out = model(x)
        pred = max(range(10), key=lambda i: out[i].data)
        correct += (pred == y)
    return correct / len(samples)

digits = load_digits()

data = [[p / 16 for p in row] for row in digits.data.tolist()]
label = digits.target.tolist()

dataset = list(zip(data, label))
random.shuffle(dataset)
split = int(0.8 * len(dataset))
train, test = dataset[:split], dataset[split:]

n = MLP(64, [16, 10])

k = 0
batch = 20
total_steps = batch * len(train)

for epoch in range(batch):
    print(f"Round: {epoch + 1} / {batch} started!")
    epoch_start = time.perf_counter()
    epoch_loss = 0
    for _data, _label in train:
        lr = 0.1 - 0.09 * k / total_steps
        k += 1

        label_pred = n(_data)
        target = [1.0 if i == _label else -1.0 for i in range(10)]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(target, label_pred))

        for p in n.parameters():
            p.grad = 0.0
        
        loss.backward()

        for p in n.parameters():
            p.data -= lr * p.grad


        if math.isnan(loss.data):
            print("Loss is nan")
            break
            
        
        epoch_loss += loss.data

    epoch_loss /= len(train)

    epoch_time = time.perf_counter() - epoch_start
    print(f"epoch {epoch + 1:2d}  loss {epoch_loss:.4f} train acc {accuracy(n, train):.3f}  test acc {accuracy(n, test):.3f} epoch time: {epoch_time:.1f}s")

for x, y in test[:5]:
    out = n(x)
    print(y, [round(o.data, 2) for o in out])
