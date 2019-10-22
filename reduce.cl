__kernel void reduce(__global const double* temperatures,
                     __local double* scratch,
                     __private int len,
                     __global double* group_sums)
{
  int gsz = get_global_size(0);
  int gix = get_global_id(0);
  int lsz = get_local_size(0);
  int lix = get_local_id(0);
  
  double acc = 0;
  for (int tix = gix; tix < len; tix+=gsz)
    acc += temperatures[tix];

  scratch[lix] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Add two halfs of the part of the temperatures array
  // that belongs to this work-group:
  for(int offset = lsz/2; offset>0; offset/=2) {
    if (lix < offset)
      scratch[lix] += scratch[lix+offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lix == 0)
    group_sums[get_group_id(0)] = scratch[0];
}
