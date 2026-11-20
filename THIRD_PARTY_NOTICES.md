# Third-party notices

## P1 / ORB-SLAM3 components

`src/m_training/native/p1/vendor/ORBextractor.cc` and `ORBextractor.h` are copied from [UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3), commit `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`. File hashes are recorded in `native/p1/UPSTREAM.json`. Their original copyright and license notices, including the OpenCV-derived notices, are retained.

P1's matching and H/F geometry code is adapted from ORB-SLAM3/ORB-SLAM2 algorithms; its existing GPL-3.0-or-later notices are retained. This distribution adapts the image-list I/O to arbitrary user filenames and fixed calibrated image sizes, without changing extraction/matching/H/F mathematics. The standalone executable communicates with Python through image lists and observation files; no SLAM System, vocabulary, mapping or loop closure is included.

The full GPL text is included at [native/p1/LICENSE-GPL-3.0.txt](src/m_training/native/p1/LICENSE-GPL-3.0.txt). The main project is also licensed under GPL-3.0-or-later. No additional noncommercial or research-only restriction is imposed.

## MLLA

The Point-MILA implementation in `src/m_training/point_mlla.py` is a point-set adaptation of [LeapLabTHU/MLLA](https://github.com/LeapLabTHU/MLLA), based on upstream commit `66b6d297e5c93ef3f8821570daf41b71b6f7782f` and an intermediate grid adaptation.

Copyright (c) Microsoft Corporation. Upstream MLLA modifications by Dongchen Han. The MIT license is reproduced in [LICENSES/MLLA-MIT.txt](LICENSES/MLLA-MIT.txt). Preserve these notices when redistributing relevant portions. The main project's GPL license does not remove the original MIT notices or the rights granted for the relevant upstream portions.

The adaptation preserves gated ELU+1 linear attention, 2D RoPE, CPE/LePE and FFN. Grid depthwise convolutions are replaced by continuous point kernels over self plus eight spatial neighbors, independently within each frame pair.

## Dependencies and data

PyTorch, NumPy, SciPy, PyYAML, Python OpenCV, native OpenCV and Eigen are external dependencies. Their own terms apply; their full distributions are not bundled. No images, KITTI data, fused labels, P1 cache, model weights or precompiled executable are included. Users must obtain appropriate rights to their own data and labels.
