# Parallel-programming-project
Accelerating forward conv layer in mxnet (ranking top 5%)

# Main optimization (see new-forward.cuh)
convolution unroll \
tile width matrix multiply \
loop unroll \
shared & constant mem \
Channel reduction \
...
