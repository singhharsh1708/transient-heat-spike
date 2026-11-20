# mosaic#375 idea 4: transient heat, Tesla T4 sweep

Peak resident memory in MB of the whole process, and in brackets its growth above the peak of a 10-step
warm-up in the same process (what the measured run needed on top of a compiled process); failures with
the peak reached. Peaks are high-water marks: growth below the warm-up peak shows as +0.


## 32 x 16 mesh (544 unknowns)

**Peak memory (MB)**

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| jax | checkpoint | 1567 (+14) | 1569 (+14) | 1564 (+13) | 1562 (+12) | 1571 (+16) |
| jax | tape | 1555 (+13) | 1556 (+9) | 1558 (+10) | 1556 (+9) | 1564 (+10) |

**Peak device memory (MB)** (the accelerator allocator's peak bytes in use; in brackets its growth over the warm-up; the allocator's limit is 14.6 GB)

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| jax | checkpoint | 96.1 (+0) | 96.1 (+0) | 96.1 (+0) | 96.1 (+0) | 96.1 (+0) |
| jax | tape | 96.1 (+0) | 96.1 (+0) | 96.1 (+0) | 160 (+64.3) | 1567 (+1471) |

**Gradient time (s)**

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| jax | checkpoint | 0.0203 | 0.0973 | 0.396 | 3.97 | 39.4 |
| jax | tape | 0.0112 | 0.0699 | 0.254 | 2.52 | 25.1 |

**Gradient along uniform conductivity, relative error against the exact gradient of the discrete scheme** (the differentiation error alone)

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| jax | checkpoint | 4.7e-15 | 6.7e-15 | 1.0e-14 | 4.2e-15 | 1.3e-14 |
| jax | tape | 4.9e-15 | 8.4e-15 | 4.4e-15 | 1.8e-14 | 4.9e-15 |

**The same against the closed form of the continuous problem** (includes the mesh's spatial error)

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| jax | checkpoint | 2.1e-03 | 6.2e-04 | 6.0e-04 | 6.0e-04 | 6.0e-04 |
| jax | tape | 2.1e-03 | 6.2e-04 | 6.0e-04 | 6.0e-04 | 6.0e-04 |

## 64 x 32 mesh (2112 unknowns)

**Peak memory (MB)**

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| jax | checkpoint | 1566 (+11) | 1558 (+5) | 1567 (+7) | 1564 (+11) | 1561 (+12) |
| jax | tape | 1553 (+4) | 1555 (+0) | 1555 (+11) | 1556 (+6) | 1555 (+7) |

**Peak device memory (MB)** (the accelerator allocator's peak bytes in use; in brackets its growth over the warm-up; the allocator's limit is 14.6 GB)

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| jax | checkpoint | 96.4 (+0) | 96.4 (+0) | 96.4 (+0) | 96.4 (+0) | 96.4 (+0) |
| jax | tape | 96.4 (+0) | 96.4 (+0) | 96.4 (+0) | 630 (+533) | 6255 (+6158) |

**Gradient time (s)**

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| jax | checkpoint | 0.0401 | 0.149 | 0.551 | 4.52 | 45.1 |
| jax | tape | 0.0225 | 0.113 | 0.357 | 3.09 | 30.6 |

**Gradient along uniform conductivity, relative error against the exact gradient of the discrete scheme** (the differentiation error alone)

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| jax | checkpoint | 3.0e-14 | 3.2e-14 | 2.8e-14 | 3.2e-14 | 3.4e-14 |
| jax | tape | 3.0e-14 | 3.3e-14 | 3.4e-14 | 4.5e-14 | 2.3e-13 |

**The same against the closed form of the continuous problem** (includes the mesh's spatial error)

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| jax | checkpoint | 1.7e-03 | 1.7e-04 | 1.5e-04 | 1.5e-04 | 1.5e-04 |
| jax | tape | 1.7e-03 | 1.7e-04 | 1.5e-04 | 1.5e-04 | 1.5e-04 |

## 128 x 64 mesh (8320 unknowns)

**Peak memory (MB)**

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| jax | checkpoint | 1564 (+15) | 1562 (+13) | 1572 (+16) | 1564 (+17) | 1561 (+15) |
| jax | tape | 1550 (+4) | 1552 (+10) | 1554 (+7) | 1556 (+9) | out of memory: needs 24.4 GB |

**Peak device memory (MB)** (the accelerator allocator's peak bytes in use; in brackets its growth over the warm-up; the allocator's limit is 14.6 GB)

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| jax | checkpoint | 97.8 (+0) | 97.7 (+0) | 97.8 (+0) | 97.8 (+0) | 97.8 (+0) |
| jax | tape | 97.8 (+0) | 97.8 (+0) | 257 (+159) | 2507 (+2409) | out of memory: needs 24.4 GB |

**Gradient time (s)**

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| jax | checkpoint | 0.0921 | 0.245 | 1.24 | 8.6 | 82.5 |
| jax | tape | 0.0709 | 0.151 | 0.763 | 5.76 | - |

**Gradient along uniform conductivity, relative error against the exact gradient of the discrete scheme** (the differentiation error alone)

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| jax | checkpoint | 1.3e-13 | 1.3e-13 | 1.3e-13 | 1.3e-13 | 1.1e-13 |
| jax | tape | 1.2e-13 | 1.3e-13 | 1.3e-13 | 1.2e-13 | - |

**The same against the closed form of the continuous problem** (includes the mesh's spatial error)

| backend | strategy | 10 steps | 100 steps | 1000 steps | 10000 steps | 100000 steps |
|---|---|---|---|---|---|---|
| jax | checkpoint | 1.5e-03 | 5.4e-05 | 3.8e-05 | 3.8e-05 | 3.8e-05 |
| jax | tape | 1.5e-03 | 5.4e-05 | 3.8e-05 | 3.8e-05 | - |

## Gradient against finite differences of the same backend's forward (16 x 8 mesh, 100 steps)

| backend | strategy | grad . d | central FD | relative difference |
|---|---|---|---|---|
| jax | checkpoint | -0.003175832244 | -0.003175832245 | 1.1e-10 |
| jax | tape | -0.003175832244 | -0.003175832245 | 2.4e-10 |
