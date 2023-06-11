# gpuMedfilt
Cuda kernel that performs fixed size windowed median filters on row major pointers.

I wrote this specifically due to MATLABs medfilt2 function being very slow for float and double matrices.

Currently the signed integer kernel versions are not working, if anyone wants to try and figure out why.

Also, it is possible to call these functions from matlab using a mex file.
