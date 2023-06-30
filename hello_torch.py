import time, torch

print(torch.__version__)
a = torch.randn(10000, 1000)
b = torch.randn(1000, 2000)
print(1)
t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print(2)
print(a.device, t1 - t0, c.norm(2))

device = torch.device('cpu')  # GPU

a = a.to(device)
b = b.to(device)

t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))

print('speedup: %.2f' % ((t1 - t0) / (t2 - t1)))

# Path: hello_torch.py