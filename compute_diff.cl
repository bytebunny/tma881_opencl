__kernel void compute_diff(__global double * diff,
                           __private double average,
                           __private int width)
{
  int ix = get_global_id(0);
  int jx = get_global_id(1);

  diff[ ix * width + jx ] -= average;
  diff[ ix * width + jx ] = ( diff[ ix * width + jx ] < 0 ? diff[ ix * width + jx ]*-1.: diff[ ix * width + jx ] );
}
