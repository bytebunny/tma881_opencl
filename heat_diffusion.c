#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "heat_diffusion.h"

int main(int argc, char* argv[])
{
  int niter=0;
  double c=0.;
  char* ptr1 = NULL;
  char* ptr2 = NULL;
  if (argc == 3 ) {
    for ( size_t ix = 1; ix < argc; ++ix ) {
      ptr1 = strchr(argv[ix], 'n' );
      ptr2 = strchr(argv[ix], 'd' );
      if ( ptr1 ){
  	niter = strtol(++ptr1, NULL, 10);
      } else if ( ptr2 ) {
  	c = atof(++ptr2);
      }
    }
  } else {
    printf("Invalid number of arguments. Correct syntax is: heat_diffusion -n#numberOfTimeSteps4 -d#diffusionConstant\n");
    exit(0);
  }

  int nx=0, ny=0;
  char line[80];
  FILE *input = fopen("diffusion", "r");
  int i=0, j=0;
  double t=0.;
  if ( input == NULL ) {
    perror("Error opening file");
    exit(0);
  }
  //read the first line
  fgets( line, sizeof(line), input);
  sscanf(line, "%d %d", &nx, &ny); // width and height.
  //store initial values (note: row major order)
  double* ivEntries = (double*) malloc( sizeof(double)*nx*ny);
  for (size_t ix = 0; ix < nx*ny; ++ix) {
    ivEntries[ix] = 0;
  }
  //read the rest
  while ( fgets(line, sizeof(line), input) != NULL) {
    sscanf(line, "%d %d %lf", &j, &i, &t);
    ivEntries[i*nx + j] = t;
  }
  fclose(input);

  ////////////////////////// Processing ///////////////////////////
  // Query and select platform:
  cl_platform_id platform_id[2]; // clinfo: 2 platforms (NVIDIA CUDA, Portable Computing Language)
  cl_uint nmb_platforms;
  if (clGetPlatformIDs( (cl_uint)2, platform_id, &nmb_platforms) != CL_SUCCESS) { 
    printf( "cannot get platform\n" );
    return(1);
  }

  // Query and select device:
  cl_device_id device_id[2]; // 2 GPUs on 1 graphic card.
  cl_uint nmb_devices;
  if (clGetDeviceIDs(platform_id[0], // graphic card is the 1st one.
                     CL_DEVICE_TYPE_GPU, nmb_platforms,
                     device_id, &nmb_devices) != CL_SUCCESS) {
    printf( "cannot get device\n" );
    return(1);
  }

  // Create context for the device:
  cl_int error_code;
  cl_context context;
  cl_context_properties properties[] = // list of context property names and their corresponding values.
  {
    CL_CONTEXT_PLATFORM, // property name.
    (cl_context_properties) platform_id[0], // property value: CUDA.
    0 // terminates the list.
  };
  context = clCreateContext( (const cl_context_properties *)properties,
                             nmb_devices, (const cl_device_id *)device_id,
                             NULL, NULL, &error_code );
  if (error_code != CL_SUCCESS) {
    printf( "cannot create context\n" );
    return(1);
  }

  // Create queue (submit task) for context and on a specific device:
  cl_command_queue command_queue;

  command_queue = clCreateCommandQueue(context,
                                       device_id[1], // use 2nd GPU in the hope that fewer people will choose it.
                                       0, &error_code);
  if (error_code != CL_SUCCESS) {
    printf("cannot create command-queue on the device 2\n");
    return(1);
  }

  // Create program object for context (from C string):
  FILE* fp;
  size_t program_size;
  fp = fopen( "compute_next_temp.cl", "r" );
  fseek( fp, 0, SEEK_END );
  program_size = ftell( fp );
  rewind( fp );
  
  // read kernel source into buffer
  char * opencl_program_src = (char*) malloc( program_size + 1 );
  opencl_program_src[ program_size ] = '\0';
  fread( opencl_program_src, sizeof(char), program_size, fp);
  fclose( fp );
  fp = NULL;

  cl_program program; // program object for a context.
  // NOTE: The devices associated with the program object are the devices associated with context.
  program = clCreateProgramWithSource( context,
                                       (cl_uint)1, // number of char buffers.
                                       (const char **) &opencl_program_src, 
                                       NULL, &error_code );
  if (error_code != CL_SUCCESS) {
    printf("cannot create program object\n");
    return(1);
  }

  // Compile and link a program executable:  
  error_code = clBuildProgram( program, 1, 
                               (const cl_device_id *)&device_id[1],
                               NULL, NULL, NULL);
  if (error_code != CL_SUCCESS) {
    printf("cannot build program. log:\n");
    
    size_t log_size = 0;
    clGetProgramBuildInfo( program, device_id[1], CL_PROGRAM_BUILD_LOG,
                           0, NULL, &log_size); // get log_size.

    char * log = calloc(log_size, sizeof(char));
    if (log == NULL) {
      printf("could not allocate memory\n");
      return(1);
    }
    clGetProgramBuildInfo( program, device_id[1], CL_PROGRAM_BUILD_LOG,
                           log_size, log, NULL); // get log info.
    printf( "%s\n", log );
    
    free(log);
    return(1);
  }

  /* Create kernel (contained in program): */
  cl_kernel kernel;
  kernel = clCreateKernel( program, (const char *)"compute_next_temp",
                           &error_code );
  if (error_code != CL_SUCCESS) {
    printf("cannot create kernel\n");
    return(1);
  }

  //Create memory objects (within context):
  const size_t total_size = nx * ny;
  cl_mem old_buffer, new_buffer;
  old_buffer  = clCreateBuffer( context, CL_MEM_READ_WRITE,
                                sizeof(cl_double) * total_size,
                                NULL, &error_code );
  if (error_code != CL_SUCCESS) {
    printf("cannot create buffer for old temperatures\n");
    return(1);
  }
  new_buffer = clCreateBuffer( context, CL_MEM_READ_WRITE,
                               sizeof(cl_double) * total_size,
                               NULL, &error_code );
  if (error_code != CL_SUCCESS) {
    printf("cannot create buffer for new temperatures\n");
    return(1);
  }

  // Copy memory objects from host to buffer object:
  cl_double* nextTimeEntries = (cl_double*) calloc(total_size, sizeof(cl_double));
  
  error_code = clEnqueueWriteBuffer( command_queue, old_buffer, CL_TRUE,
                                     0, // offset in bytes in the buffer object to write to.
                                     total_size*sizeof(cl_double),
                                     (const void *)ivEntries,
                                     0, NULL, NULL ); // events that need to to complete before this particular command can be executed.
  if (error_code != CL_SUCCESS) {
    printf("cannot enqueue writing to buffer for old temperatures\n");
    return(1);
  }

  const size_t global[] = {ny, nx};
  // Setting the pointer to local work size to NULL made program twice as slow.
  const size_t local[] = {10,10}; // CL_DEVICE_MAX_WORK_GROUP_SIZE is 32. The value has to be a divisor of global (ny and nx).
  //compute the temperatures in next time step
  cl_uint ind_old = (cl_uint)0, ind_new = (cl_uint)1, ind_dummy;
  for ( int n = 0; n < niter; ++n) {

    // Set kernel arguments:
    error_code = clSetKernelArg( kernel, ind_old,
                                 sizeof(cl_mem),
                                 (const void *)&old_buffer );
    if (error_code != CL_SUCCESS) {
      printf("cannot set kernel argument for old temperatures\n");
      return(1);
    }

    error_code = clSetKernelArg( kernel, ind_new,
                                 sizeof(cl_mem),
                                 (const void *)&new_buffer );
    if (error_code != CL_SUCCESS) {
      printf("cannot set kernel argument for new temperatures\n");
      return(1);
    }
    
    error_code = clSetKernelArg( kernel, (cl_uint)2,
                                 sizeof(double),
                                 (const void *)&c );
    if (error_code != CL_SUCCESS) {
      printf("cannot set kernel argument for conductivity\n");
      return(1);
    }

    error_code = clSetKernelArg( kernel, (cl_uint)3,
                                 sizeof(cl_uint),
                                 (const void *)&nx );
    if (error_code != CL_SUCCESS) {
      printf("cannot set kernel argument for width\n");
      return(1);
    }

    error_code = clSetKernelArg( kernel, (cl_uint)4,
                                 sizeof(cl_uint),
                                 (const void *)&ny );
    if (error_code != CL_SUCCESS) {
      printf("cannot set kernel argument for height\n");
      return(1);
    }


    // Put kernel into queue -> execution:
    error_code = clEnqueueNDRangeKernel( command_queue, kernel,
                                         (cl_uint)2, // number of dimensions used to specify the global work-items
                                         NULL, (const size_t *)&global,
                                         (const size_t *)&local, 0, NULL, NULL );
    if (error_code != CL_SUCCESS) {
      printf("cannot enqueue execution of the kernel\n");
      return(1);
    }

    // Wait until the queued commands have finished:
    error_code = clFinish(command_queue);
    if (error_code != CL_SUCCESS) {
      printf("cannot block\n");
      return(1);
    }

    // Swap old and new temperatures:
    ind_dummy = ind_old;
    ind_old = ind_new;
    ind_new = ind_dummy;
  }

  /////////////////////////// Compute average temperature //////////////////////
  const cl_int total_size_int = (cl_int)total_size;
  const size_t global_size = 1024;
  const size_t local_size = 32;
  const size_t nmb_groups = global_size / local_size;

  // Create buffer to store results of reduction:
  cl_mem output_buffer_sum;
  output_buffer_sum = clCreateBuffer( context, CL_MEM_WRITE_ONLY,
                                      sizeof(cl_double) * nmb_groups,
                                      NULL, &error_code);
  if (error_code != CL_SUCCESS) {
    printf("cannot create buffer for sums\n");
    return(1);
  }

  // Create program object for context (from C string):
  FILE* fp_reduce;
  fp_reduce = fopen( "reduce.cl", "r" );
  fseek( fp_reduce, 0, SEEK_END );
  program_size = ftell( fp_reduce );
  rewind( fp_reduce );
  
  // read kernel source into buffer
  char * opencl_program_src_reduce = (char*) malloc( program_size + 1 );
  opencl_program_src_reduce[ program_size ] = '\0';
  fread( opencl_program_src_reduce, sizeof(char), program_size, fp_reduce);
  fclose( fp_reduce );
  fp_reduce = NULL;

  cl_program program_reduction; // program object for a context.
  // NOTE: The devices associated with the program object are the devices associated with context.
  program_reduction = clCreateProgramWithSource( context,
                                                 (cl_uint)1, // number of char buffers.
                                                 (const char **) &opencl_program_src_reduce, 
                                                 NULL, &error_code );
  if (error_code != CL_SUCCESS) {
    printf("cannot create reduce program object\n");
    return(1);
  }

  // Compile and link a program executable:  
  error_code = clBuildProgram( program_reduction, (cl_uint)1, 
                               (const cl_device_id *)&device_id[1],
                               NULL, NULL, NULL);
  if (error_code != CL_SUCCESS) {
    printf("cannot build reduce program. log:\n");
    
    size_t log_size = 0;
    clGetProgramBuildInfo( program_reduction, device_id[1], CL_PROGRAM_BUILD_LOG,
                           0, NULL, &log_size); // get log_size.

    char * log = calloc(log_size, sizeof(char));
    if (log == NULL) {
      printf("could not allocate memory\n");
      return(1);
    }
    clGetProgramBuildInfo( program_reduction, device_id[1], CL_PROGRAM_BUILD_LOG,
                           log_size, log, NULL); // get log info.
    printf( "%s\n", log );
    
    free(log);
    return(1);
  }

  cl_kernel kernel_reduction;
  kernel_reduction = clCreateKernel( program_reduction, (const char *)"reduce",
                                     &error_code );
  if (error_code != CL_SUCCESS) {
    printf("cannot create reduction kernel\n");
    return(1);
  }


  if ( ind_old ) // Decide which array to copy back (must be the one whose index has value 0)
    error_code = clSetKernelArg(kernel_reduction, (cl_uint)0, sizeof(cl_mem),
                                (const void *)&new_buffer);
  else
    error_code = clSetKernelArg(kernel_reduction, (cl_uint)0, sizeof(cl_mem),
                                (const void *)&old_buffer);
  if (error_code != CL_SUCCESS) {
    printf("cannot set kernel 1st argument for reduction\n");
    return(1);
  }

  error_code = clSetKernelArg(kernel_reduction, (cl_uint)1,
                              local_size*sizeof(cl_double), NULL);
  if (error_code != CL_SUCCESS) {
    printf("cannot set kernel 2nd argument for reduction\n");
    return(1);
  }
  error_code = clSetKernelArg(kernel_reduction, (cl_uint)2,
                              sizeof(cl_int), (const void *)&total_size_int);
  if (error_code != CL_SUCCESS) {
    printf("cannot set kernel 3rd argument for reduction\n");
    return(1);
  }
  error_code = clSetKernelArg(kernel_reduction, (cl_uint)3,
                              sizeof(cl_mem), (const void *)&output_buffer_sum);
  if (error_code != CL_SUCCESS) {
    printf("cannot set kernel 4th argument for reduction\n");
    return(1);
  }

  error_code = clEnqueueNDRangeKernel( command_queue, kernel_reduction,
                                       (cl_uint)1, // number of dimensions used to specify the global work-items
                                       NULL, (const size_t *)&global_size,
                                       (const size_t *)&local_size,
                                       0, NULL, NULL );
  if (error_code != CL_SUCCESS) {
    printf("cannot enqueue execution of the reduction kernel\n");
    return(1);
  }

  // The barrier is implicit. The explicit barrier is introduced for illustration:
  error_code = clEnqueueBarrierWithWaitList(command_queue, 0, NULL, NULL);
  if (error_code != CL_SUCCESS) {
    printf("cannot enqueue a barrier\n");
    return(1);
  }

  // Copy memory objects with results from device to host
  cl_double * sums = (cl_double *)malloc( nmb_groups * sizeof(cl_double) );
  error_code = clEnqueueReadBuffer( command_queue, output_buffer_sum, CL_TRUE,
                                    0, nmb_groups * sizeof(cl_double), (void *)sums,
                                    0, NULL, NULL );
  if (error_code != CL_SUCCESS) {
    printf("cannot enqueue reading from buffer for sums\n");
    return(1);
  }

  error_code = clFinish(command_queue);
  if (error_code != CL_SUCCESS) {
    printf("cannot block\n");
    return(1);
  }

  // Perform final reduction on CPU because communication between GPU cores is much slower.
  double sum_total = 0;
  for (size_t ix=0; ix < nmb_groups; ++ix)
    sum_total += sums[ix];

  //compute average temperature
  double avg = sum_total / total_size;

  printf("average: %e\n", avg);
  

  /////////////////////// Compute abs difference with average //////////////////
  // Create program object for context (from C string):
  FILE* fp_diff;
  fp_diff = fopen( "compute_diff.cl", "r" );
  fseek( fp_diff, 0, SEEK_END );
  program_size = ftell( fp_diff );
  rewind( fp_diff );
  
  // read kernel source into buffer
  char * opencl_program_src_diff = (char*) malloc( program_size + 1 );
  opencl_program_src_diff[ program_size ] = '\0';
  fread( opencl_program_src_diff, sizeof(char), program_size, fp_diff);
  fclose( fp_diff );
  fp_diff = NULL;

  cl_program program_diff; // program object for a context.
  // NOTE: The devices associated with the program object are the devices associated with context.
  program_diff = clCreateProgramWithSource( context,
                                            (cl_uint)1, // number of char buffers.
                                            (const char **) &opencl_program_src_diff, 
                                            NULL, &error_code );
  if (error_code != CL_SUCCESS) {
    printf("cannot create difference program object\n");
    return(1);
  }

  // Compile and link a program executable:  
  error_code = clBuildProgram( program_diff, (cl_uint)1, 
                               (const cl_device_id *)&device_id[1],
                               NULL, NULL, NULL);
  if (error_code != CL_SUCCESS) {
    printf("cannot build difference program. log:\n");
    
    size_t log_size = 0;
    clGetProgramBuildInfo( program_diff, device_id[1], CL_PROGRAM_BUILD_LOG,
                           0, NULL, &log_size); // get log_size.

    char * log = calloc(log_size, sizeof(char));
    if (log == NULL) {
      printf("could not allocate memory\n");
      return(1);
    }
    clGetProgramBuildInfo( program_diff, device_id[1], CL_PROGRAM_BUILD_LOG,
                           log_size, log, NULL); // get log info.
    printf( "%s\n", log );
    
    free(log);
    return(1);
  }

  cl_kernel kernel_diff;
  kernel_diff = clCreateKernel( program_diff, (const char *)"compute_diff",
                                &error_code );
  if (error_code != CL_SUCCESS) {
    printf("cannot create difference kernel\n");
    return(1);
  }


  if ( ind_old ) // Decide which array to copy back (must be the one whose index has value 0)
    error_code = clSetKernelArg( kernel_diff, (cl_uint)0, sizeof(cl_mem),
                                 (const void *)&new_buffer);
  else
    error_code = clSetKernelArg( kernel_diff, (cl_uint)0, sizeof(cl_mem),
                                 (const void *)&old_buffer);
  if (error_code != CL_SUCCESS) {
    printf("cannot set kernel 1st argument for difference\n");
    return(1);
  }

  error_code = clSetKernelArg( kernel_diff, (cl_uint)1, sizeof(cl_double),
                               (const void *)&avg );
  if (error_code != CL_SUCCESS) {
    printf("cannot set kernel 2nd argument for difference\n");
    return(1);
  }
  error_code = clSetKernelArg( kernel_diff, (cl_uint)2, sizeof(cl_int),
                               (const void *)&nx);
  if (error_code != CL_SUCCESS) {
    printf("cannot set kernel 3rd argument for difference\n");
    return(1);
  }

  error_code = clEnqueueNDRangeKernel( command_queue, kernel_diff,
                                       (cl_uint)2, // number of dimensions used to specify the global work-items
                                       NULL, (const size_t *)&global,
                                       NULL, 0, NULL, NULL );
  if (error_code != CL_SUCCESS) {
    printf("cannot enqueue execution of the difference kernel\n");
    return(1);
  }

  // The barrier is implicit. The explicit barrier is introduced for illustration:
  error_code = clEnqueueBarrierWithWaitList(command_queue, 0, NULL, NULL);
  if (error_code != CL_SUCCESS) {
    printf("cannot enqueue a barrier\n");
    return(1);
  }

  ////////////////////// Compute average of absolute differences ///////////////
  error_code = clEnqueueNDRangeKernel( command_queue, kernel_reduction,
                                       (cl_uint)1, // number of dimensions used to specify the global work-items
                                       NULL, (const size_t *)&global_size,
                                       (const size_t *)&local_size,
                                       0, NULL, NULL );
  if (error_code != CL_SUCCESS) {
    printf("cannot enqueue 2nd execution of the reduction kernel\n");
    return(1);
  }

  // The barrier is implicit. The explicit barrier is introduced for illustration:
  error_code = clEnqueueBarrierWithWaitList(command_queue, 0, NULL, NULL);
  if (error_code != CL_SUCCESS) {
    printf("cannot enqueue a barrier after 2nd reduction\n");
    return(1);
  }

  // Copy memory objects with results from device to host
  error_code = clEnqueueReadBuffer( command_queue, output_buffer_sum, CL_TRUE,
                                    0, nmb_groups * sizeof(cl_double), (void *)sums,
                                    0, NULL, NULL );
  if (error_code != CL_SUCCESS) {
    printf("cannot enqueue 2nd reading from buffer for sums\n");
    return(1);
  }

  error_code = clFinish(command_queue);
  if (error_code != CL_SUCCESS) {
    printf("cannot block\n");
    return(1);
  }

  // Perform final reduction on CPU because communication between GPU cores is much slower.
  double diff_sum_total = 0;
  for (size_t ix=0; ix < nmb_groups; ++ix)
    diff_sum_total += sums[ix];

  //compute average temperature
  double avg_diff = diff_sum_total / total_size;

  printf("average absolute difference: %e\n", avg_diff);

  // Clean up:
  free(sums);
  free(nextTimeEntries);
  free(ivEntries);
  free(opencl_program_src);
  free(opencl_program_src_reduce);
  free(opencl_program_src_diff);

  clReleaseMemObject(new_buffer);
  clReleaseMemObject(old_buffer);
  clReleaseMemObject(output_buffer_sum);
  clReleaseKernel(kernel);
  clReleaseKernel(kernel_reduction);
  clReleaseKernel(kernel_diff);
  clReleaseProgram(program);
  clReleaseProgram(program_reduction);
  clReleaseProgram(program_diff);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);
  return 0;
}
