# fractal, 2024-2025

[![Run In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/i-a-morozov/fractal/blob/main/fractal.ipynb)

Computation of bifurcation fractals for 2D symplectic mappings

Watch supplementary videos on [YouTube](https://www.youtube.com/@sithif)

Cite as:

```
@article{fractal2025,
title = {Isochronous and period-doubling diagrams for symplectic maps of the plane},
journal = {Chaos, Solitons & Fractals},
volume = {198},
pages = {116513},
year = {2025},
issn = {0960-0779},
doi = {https://doi.org/10.1016/j.chaos.2025.116513},
url = {https://www.sciencedirect.com/science/article/pii/S0960077925005260},
author = {T. Zolkin and S. Nagaitsev and I. Morozov and S. Kladov and Y.-K. Kim},
keywords = {Chaos, Integrability, Perturbation theory, Stability}
}
```
For a quadratic mapping:

$$
\begin{aligned}
& q \to p \\
& p \to -q + w p + p^2
\end{aligned}
$$

two symmetry lines, $p=q$ and $p=1/2(w q + q^2)$  cover full 2D plane. Fractals bellow are computed in $(w, q)$ space with $w$ in [-3.00, 2.00] and $q$ in [-1.25, 1.75] using the second symmetry line. Color corresponds to different indicators.

- Phase space coverage by symmetry lines

<p align="center">
  <img width="256" height="256" src="https://github.com/i-a-morozov/fractal/blob/main/pics/orbit.gif">
</p>

- REM

<p align="center">
  <img width="576" height="256" src="https://github.com/i-a-morozov/fractal/blob/main/pics/rem.png">
</p>

- Frequency

<p align="center">
  <img width="576" height="256" src="https://github.com/i-a-morozov/fractal/blob/main/pics/frequency.png">
</p>

- Frequency & mode locking (tongues)

<p align="center">
  <img width="576" height="256" src="https://github.com/i-a-morozov/fractal/blob/main/pics/tongues.png">
</p>

- FMA

<p align="center">
  <img width="576" height="256" src="https://github.com/i-a-morozov/fractal/blob/main/pics/fma.png">
</p>

- GALI

<p align="center">
  <img width="576" height="256" src="https://github.com/i-a-morozov/fractal/blob/main/pics/gali.png">
</p>

