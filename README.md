# OtsuThresholding

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://esd100.github.io/OtsuThresholding.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://esd100.github.io/OtsuThresholding.jl/dev/)
[![Build Status](https://github.com/esd100/OtsuThresholding.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/esd100/OtsuThresholding.jl/actions/workflows/CI.yml?query=branch%3Amain)


## Origin and Acknowledgements

This package provides a Julia implementation of the multi-level Otsu thresholding algorithm, originally implemented in MATLAB by Damien Garcia (Copyright 2007-2017).

*   **Original author:** Damien Garcia
*   **Original source links:** [Author's Website](http://www.biomecardio.com/en)
*   **Reference:** Otsu N, "A Threshold Selection Method from Gray-Level Histograms," IEEE Trans. Syst. Man Cybern. 9:62-66; 1979.

This Julia version is a translation and adaptation of the original MATLAB code. Translation and refactoring assistance was provided by AI (Google Gemini 2.5 Pro Experimental 03-25).


## Installation

The package can be installed using the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add OtsuThresholding
```

Or, just run:

```julia
using Pkg
Pkg.add("OtsuThresholding")
```

## Basic Usage

The main function exported is otsu. It takes an image and an optional number of classes n (defaulting to 2 for standard binary thresholding) and returns an index map and a separability measure.

```julia

using OtsuThresholding
using TestImages, Images, ImageView # For example images and viewing

# --- Grayscale Example ---
img_gray = testimage("mandril_gray")

# Binary thresholding (n=2)
IDX2, sep2 = otsu(img_gray)
println("Separability (n=2): ", sep2)
# imshow(IDX2) # View the resulting 2-class index map

# Multi-level thresholding (n=4)
IDX4, sep4 = otsu(img_gray, 4)
println("Separability (n=4): ", sep4)
# imshow(IDX4) # View the resulting 4-class index map

# --- RGB Example ---
# The function automatically converts RGB using KLT/PCA
img_rgb = testimage("mandrill")

IDX_rgb, sep_rgb = otsu(img_rgb, 3)
println("Separability for RGB (n=3): ", sep_rgb)
# imshow(IDX_rgb) # View the resulting 3-class index map
```

The returned IDX is a Matrix{Int} where each element is the class index (from 1 to n) assigned to the corresponding pixel. Pixels that were non-finite (NaN or Inf) in the original input are assigned 0.

The returned sep is a Float64 value between 0.0 and 1.0 indicating the quality of the separation according to Otsu's criterion (normalized between-class variance).

## License
This package is licensed under the MIT License. See the LICENSE file for details. Note the included copyright notice for the original MATLAB implementation by Damien Garcia.