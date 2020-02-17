PROOF
=====

``find_filaments_ridges.py`` is the main file of interest.
It is a script which runs over an image and tries to extract the vector paths of the filaments.

Originally it was written to run over the raw micrographs but now should be run over the output of the segmentation network.

In summary it:

- Extracts line segments using Hough transforms
- Merges line segments using LSM
- Traces segments using ``trace_filament.py``

LSM
---

``lsm.py`` contains a Python implementation of a published algorithm for merging line segments.

Filament Tracing
----------------

``trace_filament.py`` is a custom algorithm to try to follow along filaments and make a poly-line for each filament.
