__kernel void compute_next_temp(__global const double * old_temp,
                                __global double * new_temp,
                                __private double conductivity,
                                __private int width,
                                __private int height)
{
  int ix = get_global_id(0);
  int jx = get_global_id(1);

  double hij, hijW, hijE, hijS, hijN;
  hij = old_temp[ ix * width + jx ];
  hijW = ( jx-1 >= 0 ? old_temp[ix*width + jx-1] : 0. );
  hijE = ( jx+1 < width ? old_temp[ix*width + jx+1] : 0.);
  hijS = ( ix+1 < height ? old_temp[(ix+1)*width + jx] : 0.);
  hijN = ( ix-1 >= 0 ? old_temp[(ix-1)*width + jx] : 0.);

  new_temp[ ix * width + jx ] = hij + conductivity * 
     ( 0.25 * ( hijW + hijE + hijS + hijN ) - hij );
  //printf("%d, %d: %lf\n",ix,jx, new_temp[ix*width + jx]);
}
