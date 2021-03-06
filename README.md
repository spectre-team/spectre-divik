[![CodeFactor](https://www.codefactor.io/repository/github/spectre-team/spectre-divik/badge)](https://www.codefactor.io/repository/github/spectre-team/spectre-divik)
[![BCH compliance](https://bettercodehub.com/edge/badge/spectre-team/spectre-divik?branch=master)](https://bettercodehub.com/)
[![Maintainability](https://api.codeclimate.com/v1/badges/12bf3a9343ab563e2b89/maintainability)](https://codeclimate.com/github/spectre-team/spectre-divik/maintainability)
[![Windows build status](https://ci.appveyor.com/api/projects/status/962q796vgnve968u/branch/master?svg=true)](https://ci.appveyor.com/project/gmrukwa/spectre-divik/branch/master)
[![Linux build Status](https://travis-ci.org/spectre-team/spectre-divik.svg?branch=master)](https://travis-ci.org/spectre-team/spectre-divik)

![Spectre](https://user-images.githubusercontent.com/1897842/31115297-0fe2c3aa-a822-11e7-90e6-92ceccf76137.jpg)

# spectre-divik

Python implementation of Divisive iK-means (DiviK) algorithm.

# Tools within this package

> This section will be further developed soon.

1) [`divik`](./spdivik/README.md) - runs DiviK in one of many scenarios
2) [`kmeans`](./spdivik/kmeans/README.md) - runs K-means
3) `linkage` - runs agglomerative clustering
4) [`inspect`](./spdivik/inspect/README.md) - visualizes DiviK result
5) `visualize` - generates `.png` file with visualization of clusters
6) [`spectral`](./spdivik/spectral.md) - generates spectral embedding of a
dataset

# Installation

## Docker

The recommended way to use this software is through
[Docker](https://www.docker.com/). This is the most convenient way, if you want
to use `divik` application, since it requires *MATLAB Compiler Runtime*
and more dependencies.

To install latest stable version use:

```bash
docker pull gmrukwa/divik
```

To install specific version, you can specify it in the command, e.g.:

```bash
docker pull gmrukwa/divik:1.12.0
```

## Python package

Prerequisites for installation of base package:

- Python 3.5
- [functional helpers](https://github.com/gmrukwa/functional-helpers)

These are required for using `divik` application:

- [MATLAB Compiler Runtime](https://www.mathworks.com/products/compiler/matlab-runtime.html),
version 2016b or newer, installed to default path
- [compiled package with legacy code](https://github.com/spectre-team/matlab-legacy/releases/tag/legacy-v4.0.9)

Installation process may be clearer with insight into Docker images used for
application deployment:

- [`python_mcr` image](https://github.com/spectre-team/python_mcr) - installs
MCR r2016b onto Python 3.5 image
- [`python_msi` image](https://github.com/spectre-team/python_msi) - installs
compiled legacy code onto MCR image
- [`divik` image](https://github.com/spectre-team/spectre-divik/blob/master/dockerfile) -
installs DiviK software onto legacy code image

Functional helpers should be installed with:

```bash
pip install git+https://github.com/gmrukwa/functional-helpers.git@2e68a8801f894a14601d70db76086ada723bac35#egg=functional_helpers
```

Having prerequisites installed, one can install latest base version of the
package:

```bash
pip install git+https://github.com/spectre-team/spectre-divik.git@master#egg=spectre-divik
```

or any stable tagged version, e.g.:

```bash
pip install git+https://github.com/spectre-team/spectre-divik.git@v1.12.0#egg=spectre-divik
```

Installation of `divik` program dependencies can be validated via:

```bash
pip install git+https://github.com/spectre-team/spectre-divik.git@master#egg=spectre-divik[divik]
```

**Note:** *Using zsh you may need to escape square brackets with `\ `*

If you want to take advantage of using [Quilt](https://quiltdata.com) for data
management, you can install also this extra:

```bash
pip install git+https://github.com/spectre-team/spectre-divik.git@master#egg=spectre-divik[quilt_packages]
```

# References

This software is part of contribution made by [Data Mining Group of Silesian
University of Technology](http://www.zaed.polsl.pl/), rest of which is
published [here](https://github.com/ZAEDPolSl).

+ [P. Widlak, G. Mrukwa, M. Kalinowska, M. Pietrowska, M. Chekan, J. Wierzgon, M.
Gawin, G. Drazek and J. Polanska, "Detection of molecular signatures of oral
squamous cell carcinoma and normal epithelium - application of a novel
methodology for unsupervised segmentation of imaging mass spectrometry data,"
Proteomics, vol. 16, no. 11-12, pp. 1613-21, 2016][1]
+ [M. Pietrowska, H. C. Diehl, G. Mrukwa, M. Kalinowska-Herok, M. Gawin, M.
Chekan, J. Elm, G. Drazek, A. Krawczyk, D. Lange, H. E. Meyer, J. Polanska, C.
Henkel, P. Widlak, "Molecular profiles of thyroid cancer subtypes:
Classification based on features of tissue revealed by mass spectrometry
imaging," Biochimica et Biophysica Acta (BBA)-Proteins and Proteomics, 2016][2]
+ [G. Mrukwa, G. Drazek, M. Pietrowska, P. Widlak and J. Polanska, "A Novel
Divisive iK-Means Algorithm with Region-Driven Feature Selection as a Tool for
Automated Detection of Tumour Heterogeneity in MALDI IMS Experiments," in
International Conference on Bioinformatics and Biomedical Engineering, 2016][3]

[1]: http://onlinelibrary.wiley.com/doi/10.1002/pmic.201500458/pdf
[2]: http://www.sciencedirect.com/science/article/pii/S1570963916302175
[3]: http://link.springer.com/chapter/10.1007/978-3-319-31744-1_11
