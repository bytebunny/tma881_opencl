__kernel void dot_prod_mul(__global const float * a,
                           __global const float * b,
                           __global float * c
                           )
{
  int ix = get_global_id(0);
  c[ix] = a[ix] * b[ix];
}
