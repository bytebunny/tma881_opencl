[//]: # (To preview markdown file in Emacs type C-c C-c p)

# [Assignment 4: OpenCL](https://www.raum-brothers.eu/martin/Chalmers_TMA881_1920/assignments.html#opencl)
> A simple model for heat diffiusion in 2-dimensional space.

A typical simple OpenCL program presented in lectures is summarised in `test_opencl.c` (which in turn uses a kernel located in `dot_prod_mul.cl`). This program does not present **reduction**, which is only used in the full program `heat_diffusion.c`.

**NOTE:** the program is configured to use the 2nd (out of 2) device (GPU) of the 1st (out of 2) platform (graphics card) on Gantenbein, so either this should be adjusted to the system the program is built on, or it the system should have at least 2 devices on
the 1st platform.
