# Imports & definitions

import numpy
import numba

from matplotlib import pyplot as plt
from matplotlib import colormaps

from tqdm import tqdm

fastmath:bool=False
parallel:bool=True

@numba.jit('float64[:](int64, float64)', nopython=True, fastmath=fastmath)
def window(n, s):
    t = numpy.linspace(0.0, (n - 1.0)/n, n)
    f = numpy.exp(-1.0/((1.0 - t)**s*t**s))
    return f/numpy.sum(f)

@numba.jit('Tuple((float64[:], float64[:]))(float64, float64[:], float64[:])', nopython=True, fastmath=fastmath)
def mapping(w, q, p):
    return p, -q + w*p + p**2

@numba.jit('Tuple((float64, float64))(float64, float64, float64)', nopython=True, fastmath=fastmath)
def forward(w, q, p):
    return p, -q + w*p + p**2

@numba.jit('Tuple((float64, float64))(float64, float64, float64)', nopython=True, fastmath=fastmath)
def inverse(w, q, p):
    return -p + w*q + q**2, q

@numba.jit('Tuple((float64[:, :], float64[:, :]))(int64, float64, float64[:], float64[:])', nopython=True, parallel=parallel, fastmath=fastmath)
def orbit(n, w, q, p):
    qs = numpy.zeros((n + 1, len(q)))
    ps = numpy.zeros((n + 1, len(p)))
    qs[0], ps[0] = q, p
    for i in range(1, n + 1):
        qs[i], ps[i] = q, p = mapping(w, q, p)
    return qs.T, ps.T

@numba.jit('float64[:](int64, float64, float64[:, :])', nopython=True, parallel=parallel, fastmath=fastmath)
def rem(n, w, qp):
    out = numpy.zeros(len(qp))
    for i in numba.prange(len(qp)):
        q, p = qp[i]
        Q, P = q, p
        for _ in range(n):
            Q, P = forward(w, Q, P)
        Q, P = Q + 2.5E-16, P + 2.5E-16
        for _ in range(n):
            Q, P = inverse(w, Q, P)
        out[i] = numpy.log10(1.0E-15 + numpy.sqrt((q - Q)**2 + (p - P)**2))
    return out

q = numpy.linspace(-1.25, 1.74, 5001)
p = numpy.linspace(-1.25, 1.75, 5001)

qp = numpy.stack(numpy.meshgrid(q, p, indexing='ij')).swapaxes(-1, 0).reshape(5001*5001, -1)

cmap = colormaps.get_cmap('viridis')
cmap.set_bad(color='lightgray')

ws =  numpy.linspace(2.0, -3.0, 2401)
dw = 0.0

for i, w in tqdm(enumerate(ws)):

    out = rem(2**12, w+dw, qp)
    out = out.reshape(5001, 5001)
    out[out > 0.0] = 0.0
    out[out < - 16.0] = -16.0

    plt.figure(figsize=(8, 8))
    plt.imshow(out, aspect='equal', vmin=-15.0, vmax=0.0, origin='lower', cmap=cmap, interpolation='nearest', extent=(-1.25, 1.75, -1.25, 1.75))
    plt.title(f'w={w+dw:.3f}')
    plt.xlabel('Q')
    plt.ylabel('P')
    plt.tight_layout()
    plt.savefig(f'{i:04}_dw_fractal.png', dpi=300)
    plt.close()
