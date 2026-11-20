# Third-party notices

## ORB frontend

`ORBextractor.cc` and `ORBextractor.h` come from [UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3), commit `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`. File hashes are in [UPSTREAM.json](src/m_training/native/p1/UPSTREAM.json). The matching and H/F geometry adapters derive from ORB-SLAM3/ORB-SLAM2. Original copyright, GPL, and OpenCV notices remain in the source files.

License: [GPL-3.0-or-later](src/m_training/native/p1/LICENSE-GPL-3.0.txt).

## MLLA

`src/m_training/point_mlla.py` adapts [LeapLabTHU/MLLA](https://github.com/LeapLabTHU/MLLA), commit `66b6d297e5c93ef3f8821570daf41b71b6f7782f`, through an intermediate grid implementation. Grid depthwise convolutions are replaced by point kernels over self and eight spatial neighbors.

Copyright (c) Microsoft Corporation. Upstream MLLA modifications by Dongchen Han. License: [MIT](LICENSES/MLLA-MIT.txt). These notices apply to the relevant upstream portions.

## Dependencies

PyTorch, NumPy, SciPy, PyYAML, OpenCV, and Eigen are external dependencies under their respective licenses.
