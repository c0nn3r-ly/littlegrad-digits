import random
import math
import time
from sklearn.datasets import load_digits
from micrograd.littlegrad.nn import MLP
import json, pathlib, datetime

def accuracy(model, samples):
    correct = 0
    for x, y in samples:
        out = model(x)
        pred = max(range(10), key=lambda i: out[i].data)
        correct += (pred == y)
    return correct / len(samples)

sizes = "-".join(str(s) for s in [64] + [32, 32, 10])
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

log_dir = pathlib.Path(__file__).resolve().parent / "logs"
log_dir.mkdir(exist_ok=True)

random.seed(0)

start = time.perf_counter()

digits = load_digits()

data = [[p / 16 for p in row] for row in digits.data.tolist()]
label = digits.target.tolist()

dataset = list(zip(data, label))
random.shuffle(dataset)
split = int(0.8 * len(dataset))
train, test = dataset[:split], dataset[split:]

n = MLP(64, [32, 32, 10]) #init a model from scratch
""" open a model
with open("logs/xxx.json") as f:
    r = json.load(f)
n = MLP(r["nin"], r["nouts"])
for p, w in zip(n.parameters(), r["weights"]):
    p.data = w
"""
k = 0
batch = 30
total_steps = batch * len(train)

for epoch in range(batch):
    print(f"Round: {epoch + 1} / {batch} started!")
    epoch_start = time.perf_counter()
    epoch_loss = 0
    for _data, _label in train:
        lr = 0.03 - 0.029 * k / total_steps
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

result = {
    "nin": 64,
    "nouts": [32, 32, 10],
    "epochs": batch,
    "final_loss": epoch_loss,
    "train_acc": accuracy(n, train),
    "test_acc": accuracy(n, test),
    "weights": [p.data for p in n.parameters()],
}
print("Saving results...")
with open(log_dir / f"{stamp}_{sizes}.json", "w") as f:
    json.dump(result, f)
print("Done!")