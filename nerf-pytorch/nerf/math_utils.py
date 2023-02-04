import torch
import numpy as np

EPS = torch.tensor(1e-7)

def saturate(x, low=0.0, high=1.0):
    return torch.clip(x, low, high)

def srgb_to_linear(x):
    x = saturate(x)

    switch_val = torch.tensor(0.04045)
    return torch.where(
        torch.ge(x, switch_val),
        torch.pow((torch.maximum(x, switch_val) + 0.055) / 1.055, 2.4),
        x / 12.92,
    )

def mix(x, y, a):
    a = torch.clip(a, 0, 1)
    return x * (1 - a) + y * a

def magnitude(x):
    return safe_sqrt(dot(x, x))

def safe_sqrt(x):
    sqrt_in = torch.maximum(x, EPS)
    return torch.sqrt(sqrt_in)

def normalize(x):
    magn = magnitude(x)
    return torch.where(magn <= safe_sqrt(torch.tensor(0)), torch.zeros_like(x), x / magn)

def dot(x, y):
    return torch.sum(x * y, axis=-1, keepdims=True)

def safe_exp(x):
    return torch.exp(torch.minimum(x, fill_like(x, 87.5)))

def to_vec3(x):
    return repeat(x, 3, -1)

def repeat(x, n, axis):
    repeat = [1 for _ in range(len(x.shape))]
    repeat[axis] = n

    return torch.tile(x, repeat)

def safe_log(x):
    return torch.log(torch.minimum(x, fill_like(x, 33e37)))

def fill_like(x, val):
    return torch.ones_like(x) * val

def reflect(d, n):
    return d - 2 * dot(d, n) * n