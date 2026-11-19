# mosaic#375 idea 4: transient heat, CPU sweep

Peak resident memory in MB of the whole process, and in brackets its growth above the peak of a 10-step
warm-up in the same process (what the measured run needed on top of a compiled process); failures with
the peak reached. Peaks are high-water marks: growth below the warm-up peak shows as +0.


## 32 x 16 mesh (544 unknowns)

**Peak memory (MB)**

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| firedrake | checkpoint | 289 (+4) | 293 (+9) | 307 (+23) | 369 (+84) | 941 (+657) |
| firedrake | tape | 289 (+4) | 293 (+9) | 330 (+46) | 732 (+448) | 4748 (+4465) |
| jax | checkpoint | 312 (+5) | 317 (+9) | 323 (+15) | 329 (+22) | 337 (+30) |
| jax | tape | 309 (+6) | 312 (+8) | 340 (+37) | 478 (+177) | 1890 (+1587) |
| julia | backsolve | 1513 (+0) | 1501 (+0) | 1445 (+0) | 1709 (+0) | 1691 (+0) |
| julia | gauss | 1646 (+0) | 1745 (+0) | 1630 (+0) | 1657 (+12) | 3354 (+1689) |
| julia | interp | 1590 (+0) | 1552 (+0) | 1615 (+0) | 1580 (+53) | 1577 (+26) |
| julia | interp-dense | 1571 (+0) | 1571 (+0) | 1613 (+0) | 1621 (+74) | 3235 (+1676) |
| julia | quadrature | 1733 (+0) | 1646 (+0) | 1530 (+0) | 1813 (+151) | 5073 (+3373) |

**Gradient time (s)**

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| firedrake | checkpoint | 0.164 | 1.52 | 15.4 | 155 | 1.55e+03 |
| firedrake | tape | 0.061 | 0.438 | 4.41 | 44.6 | 477 |
| jax | checkpoint | 0.00406 | 0.0159 | 0.114 | 1.22 | 12.1 |
| jax | tape | 0.00165 | 0.00988 | 0.0982 | 0.787 | 7.49 |
| julia | backsolve | 1.01 | 1.01 | 1.06 | 3.2 | 31.6 |
| julia | gauss | 1.28 | 1.33 | 2.12 | 10 | 84.7 |
| julia | interp | 1.32 | 1.46 | 2.7 | 14.4 | 128 |
| julia | interp-dense | 1.21 | 1.37 | 2.85 | 18.9 | 225 |
| julia | quadrature | 0.935 | 0.976 | 1.6 | 8.14 | 72.1 |

**Gradient along uniform conductivity, relative error against the exact gradient of the discrete scheme** (the differentiation error alone)

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| firedrake | checkpoint | 1.1e-14 | 3.6e-14 | 7.4e-15 | 1.4e-12 | 2.1e-11 |
| firedrake | tape | 1.1e-14 | 3.6e-14 | 7.4e-15 | 1.4e-12 | 2.1e-11 |
| jax | checkpoint | 2.2e-14 | 4.2e-14 | 5.1e-15 | 1.3e-12 | 2.1e-11 |
| jax | tape | 2.2e-14 | 4.2e-14 | 3.9e-15 | 1.3e-12 | 2.1e-11 |
| julia | backsolve | 8.9e-01 | 9.8e-01 | 9.9e-01 | 1.0e+00 | 5.3e+289 |
| julia | gauss | 1.0e-01 | 1.0e-02 | 1.0e-03 | 1.0e-04 | 1.0e-05 |
| julia | interp | 9.5e-02 | 1.0e-02 | 2.1e-03 | 2.1e-04 | 2.1e-05 |
| julia | interp-dense | 1.0e-01 | 1.0e-02 | 2.0e-03 | 2.0e-04 | 2.0e-05 |
| julia | quadrature | 1.0e-01 | 1.0e-02 | 1.0e-03 | 1.0e-04 | 1.0e-05 |

**The same against the closed form of the continuous problem** (includes the mesh's spatial error)

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| firedrake | checkpoint | 2.1e-03 | 6.2e-04 | 6.0e-04 | 6.0e-04 | 6.0e-04 |
| firedrake | tape | 2.1e-03 | 6.2e-04 | 6.0e-04 | 6.0e-04 | 6.0e-04 |
| jax | checkpoint | 2.1e-03 | 6.2e-04 | 6.0e-04 | 6.0e-04 | 6.0e-04 |
| jax | tape | 2.1e-03 | 6.2e-04 | 6.0e-04 | 6.0e-04 | 6.0e-04 |
| julia | backsolve | 8.9e-01 | 9.8e-01 | 9.9e-01 | 1.0e+00 | 5.3e+289 |
| julia | gauss | 1.0e-01 | 9.4e-03 | 4.0e-04 | 5.0e-04 | 5.9e-04 |
| julia | interp | 9.3e-02 | 9.7e-03 | 1.5e-03 | 3.9e-04 | 5.8e-04 |
| julia | interp-dense | 9.8e-02 | 9.4e-03 | 1.4e-03 | 4.0e-04 | 5.8e-04 |
| julia | quadrature | 9.9e-02 | 9.4e-03 | 4.0e-04 | 5.0e-04 | 5.9e-04 |

## 64 x 32 mesh (2112 unknowns)

**Peak memory (MB)**

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| firedrake | checkpoint | 311 (+5) | 301 (+12) | 330 (+40) | 408 (+119) | 1145 (+855) |
| firedrake | tape | 310 (+5) | 302 (+14) | 382 (+94) | 1218 (+930) | 9533 (+9244) |
| jax | checkpoint | 326 (+9) | 335 (+18) | 336 (+23) | 343 (+29) | 358 (+40) |
| jax | tape | 313 (+7) | 317 (+15) | 376 (+75) | 945 (+644) | 6570 (+6270) |
| julia | backsolve | 1469 (+0) | 1494 (+0) | 1528 (+0) | 1532 (+0) | 1463 (+0) |
| julia | gauss | 1668 (+0) | 1519 (+0) | 1666 (+0) | 2579 (+887) | 6854 (+5402) |
| julia | interp | 1651 (+0) | 1701 (+0) | 1626 (+0) | 1663 (+15) | 1644 (+0) |
| julia | interp-dense | 1628 (+0) | 1639 (+0) | 1896 (+0) | 2403 (+884) | 6812 (+5183) |
| julia | quadrature | 1551 (+0) | 1566 (+0) | 1775 (+0) | 2806 (+1234) | 11668 (+10122) |

**Gradient time (s)**

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| firedrake | checkpoint | 0.151 | 1.31 | 13.6 | 140 | 1.29e+03 |
| firedrake | tape | 0.0893 | 0.615 | 5.2 | 56.7 | 584 |
| jax | checkpoint | 0.0149 | 0.0792 | 0.553 | 4.76 | 48 |
| jax | tape | 0.00653 | 0.039 | 0.294 | 2.73 | 27 |
| julia | backsolve | 59.2 | 59.3 | 51.7 | 67.8 | 239 |
| julia | gauss | 63.1 | 58.6 | 72.5 | 105 | 504 |
| julia | interp | 70.3 | 70.9 | 76.4 | 133 | 692 |
| julia | interp-dense | 70.5 | 71.1 | 77.8 | 149 | 800 |
| julia | quadrature | 57.8 | 56 | 60.2 | 92.8 | 507 |

**Gradient along uniform conductivity, relative error against the exact gradient of the discrete scheme** (the differentiation error alone)

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| firedrake | checkpoint | 1.0e-13 | 1.0e-13 | 1.1e-13 | 1.7e-12 | 1.1e-11 |
| firedrake | tape | 1.0e-13 | 1.0e-13 | 1.1e-13 | 1.7e-12 | 1.1e-11 |
| jax | checkpoint | 5.5e-14 | 8.3e-14 | 8.4e-14 | 1.7e-12 | 1.2e-11 |
| jax | tape | 5.5e-14 | 8.3e-14 | 8.4e-14 | 1.7e-12 | 1.1e-11 |
| julia | backsolve | 1.0e+00 | 9.8e-01 | 1.0e+00 | 1.0e+00 | 1.0e+00 |
| julia | gauss | 1.0e-01 | 1.0e-02 | 1.0e-03 | 1.0e-04 | 1.0e-05 |
| julia | interp | 9.5e-02 | 1.4e-02 | 2.1e-03 | 2.1e-04 | 2.1e-05 |
| julia | interp-dense | 1.0e-01 | 1.3e-02 | 2.0e-03 | 2.0e-04 | 2.0e-05 |
| julia | quadrature | 1.0e-01 | 1.0e-02 | 1.0e-03 | 1.0e-04 | 1.0e-05 |

**The same against the closed form of the continuous problem** (includes the mesh's spatial error)

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| firedrake | checkpoint | 1.7e-03 | 1.7e-04 | 1.5e-04 | 1.5e-04 | 1.5e-04 |
| firedrake | tape | 1.7e-03 | 1.7e-04 | 1.5e-04 | 1.5e-04 | 1.5e-04 |
| jax | checkpoint | 1.7e-03 | 1.7e-04 | 1.5e-04 | 1.5e-04 | 1.5e-04 |
| jax | tape | 1.7e-03 | 1.7e-04 | 1.5e-04 | 1.5e-04 | 1.5e-04 |
| julia | backsolve | 1.0e+00 | 9.8e-01 | 1.0e+00 | 1.0e+00 | 1.0e+00 |
| julia | gauss | 1.0e-01 | 9.9e-03 | 8.5e-04 | 5.1e-05 | 1.4e-04 |
| julia | interp | 9.3e-02 | 1.3e-02 | 2.0e-03 | 6.2e-05 | 1.3e-04 |
| julia | interp-dense | 9.8e-02 | 1.3e-02 | 1.8e-03 | 4.9e-05 | 1.3e-04 |
| julia | quadrature | 9.9e-02 | 9.8e-03 | 8.5e-04 | 5.1e-05 | 1.4e-04 |

## 128 x 64 mesh (8320 unknowns)

**Peak memory (MB)**

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| firedrake | checkpoint | 336 (+9) | 337 (+27) | 420 (+109) | 568 (+257) | 1935 (+1625) |
| firedrake | tape | 330 (+10) | 339 (+36) | 585 (+282) | 3133 (+2831) | killed at 13.0 GB |
| jax | checkpoint | 335 (+14) | 333 (+17) | 335 (+21) | 361 (+49) | 416 (+102) |
| jax | tape | 330 (+13) | 349 (+43) | 574 (+267) | 2824 (+2517) | out of memory: needs 24.4 GB |

**Gradient time (s)**

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| firedrake | checkpoint | 0.257 | 1.67 | 16 | 173 | 1.71e+03 |
| firedrake | tape | 0.167 | 0.801 | 7.33 | 77 | - |
| jax | checkpoint | 0.0902 | 0.352 | 1.9 | 12.4 | 118 |
| jax | tape | 0.0591 | 0.262 | 1.61 | 11.7 | - |

**Gradient along uniform conductivity, relative error against the exact gradient of the discrete scheme** (the differentiation error alone)

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| firedrake | checkpoint | 1.3e-12 | 1.5e-12 | 1.4e-12 | 3.6e-12 | 1.7e-11 |
| firedrake | tape | 1.3e-12 | 1.5e-12 | 1.4e-12 | 3.6e-12 | - |
| jax | checkpoint | 1.5e-12 | 1.7e-12 | 1.5e-12 | 3.7e-12 | 1.7e-11 |
| jax | tape | 1.5e-12 | 1.7e-12 | 1.5e-12 | 3.7e-12 | - |

**The same against the closed form of the continuous problem** (includes the mesh's spatial error)

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| firedrake | checkpoint | 1.5e-03 | 5.4e-05 | 3.8e-05 | 3.8e-05 | 3.8e-05 |
| firedrake | tape | 1.5e-03 | 5.4e-05 | 3.8e-05 | 3.8e-05 | - |
| jax | checkpoint | 1.5e-03 | 5.4e-05 | 3.8e-05 | 3.8e-05 | 3.8e-05 |
| jax | tape | 1.5e-03 | 5.4e-05 | 3.8e-05 | 3.8e-05 | - |

## Gradient against finite differences of the same backend's forward (16 x 8 mesh, 100 steps)

| backend | strategy | grad . d | central FD | relative difference |
|---|---|---|---|---|
| firedrake | checkpoint | -0.001525646116 | -0.001525646115 | 8.8e-11 |
| firedrake | tape | -0.001525646116 | -0.001525646115 | 8.8e-11 |
| jax | checkpoint | -0.003175832244 | -0.003175832244 | 1.1e-10 |
| jax | tape | -0.003175832244 | -0.003175832244 | 1.1e-10 |
| julia | backsolve | 4.705381152e-05 | 0.001164484154 | 9.6e-01 |
| julia | gauss | 0.001176167511 | 0.001164484154 | 1.0e-02 |
| julia | interp | 0.001176558385 | 0.001164484154 | 1.0e-02 |
| julia | interp-dense | 0.001176138352 | 0.001164484154 | 1.0e-02 |
| julia | quadrature | 0.001176158675 | 0.001164484154 | 1.0e-02 |

## k(T) follow-on: Newton per step, unrolled vs implicit differentiation (JAX, 16 x 8 mesh)

Errors against the converged implicit gradient (30 Newton iterations, its own process; against central
differences: 10 steps 4e-09; 50 steps 3e-10; 200 steps 1e-09). Peak MB is the whole process's (dense Newton solves).

| steps | Newton iterations | forward J error | unrolled: gradient error, peak MB, s | implicit: gradient error, peak MB, s |
|---|---|---|---|---|
| 10 | 1 | 5.8e-03 | 6.5e-04, 324, 0.009 | 4.6e-03, 321, 0.010 |
| 10 | 2 | 1.5e-07 | 1.5e-08, 350, 0.017 | 7.5e-08, 324, 0.016 |
| 10 | 3 | 2.2e-15 | 2.1e-14, 356, 0.025 | 6.7e-16, 321, 0.019 |
| 10 | 5 | 1.9e-15 | 5.6e-16, 360, 0.041 | 2.7e-15, 325, 0.027 |
| 50 | 1 | 2.6e-04 | 1.2e-05, 344, 0.041 | 2.1e-04, 325, 0.047 |
| 50 | 2 | 4.3e-11 | 1.3e-10, 366, 0.080 | 3.8e-11, 326, 0.071 |
| 50 | 3 | 1.9e-15 | 1.1e-15, 381, 0.125 | 7.8e-16, 327, 0.090 |
| 50 | 5 | 0.0e+00 | 2.2e-15, 404, 0.203 | 1.1e-15, 328, 0.126 |
| 200 | 1 | 1.7e-05 | 5.8e-07, 375, 0.161 | 1.3e-05, 331, 0.190 |
| 200 | 2 | 1.8e-14 | 8.8e-14, 425, 0.319 | 1.1e-14, 332, 0.284 |
| 200 | 3 | 0.0e+00 | 2.9e-15, 474, 0.487 | 2.2e-15, 338, 0.353 |
| 200 | 5 | 0.0e+00 | 2.2e-15, 552, 0.792 | 2.0e-15, 339, 0.501 |
