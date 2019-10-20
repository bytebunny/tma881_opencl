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
    sscanf(line, "%d %d %lf", &i, &j, &t);
    ivEntries[i*ny + j] = t;
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

  /* // Create kernel (contained in program): */
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
                                 &c );
    if (error_code != CL_SUCCESS) {
      printf("cannot set kernel argument for conductivity\n");
      return(1);
    }

    error_code = clSetKernelArg( kernel, (cl_uint)3,
                                 sizeof(cl_uint),
                                 &nx );
    if (error_code != CL_SUCCESS) {
      printf("cannot set kernel argument for width\n");
      return(1);
    }

    error_code = clSetKernelArg( kernel, (cl_uint)4,
                                 sizeof(cl_uint),
                                 &ny );
    if (error_code != CL_SUCCESS) {
      printf("cannot set kernel argument for height\n");
      return(1);
    }


    // Put kernel into queue -> execution:
    const size_t global[] = {nx, ny};
    error_code = clEnqueueNDRangeKernel( command_queue, kernel,
                                         (cl_uint)2, // number of dimensions used to specify the global work-items
                                         NULL, (const size_t *)&global,
                                         NULL, 0, NULL, NULL );
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

  // Copy memory objects with results from device to host  
  if ( ind_old ) // Decide which array to copy back (must be the one whose index has value 0)
    error_code = clEnqueueReadBuffer( command_queue, new_buffer, CL_TRUE,
                                      0, total_size*sizeof(cl_double),
                                      (void *)nextTimeEntries,
                                      0, NULL, NULL );
  else 
    error_code = clEnqueueReadBuffer( command_queue, old_buffer, CL_TRUE,
                                      0, total_size*sizeof(cl_double),
                                      (void *)nextTimeEntries,
                                      0, NULL, NULL );
  if (error_code != CL_SUCCESS) {
    printf("cannot enqueue reading from buffer\n");
    return(1);
  }




  //compute average temperature
  double avg = computeAverage(nextTimeEntries, nx, ny);

  double* avgDiffEntries = (double*) malloc(sizeof(double)*nx*ny);
  for (int ix = 0; ix < nx*ny; ++ix) {
    avgDiffEntries[ix] = 0.;
  }

  /* //compute differences (in nextTimeEntries since the contents are copied to ivEntries) */
  if ( niter != 0) {
    memcpy(avgDiffEntries, nextTimeEntries, nx*ny*sizeof(double));
  } else {
    memcpy(avgDiffEntries, ivEntries, nx*ny*sizeof(double));
  }
  for ( int ix = 0; ix < nx*ny; ++ix ) {
    avgDiffEntries[ix] -= avg;
    avgDiffEntries[ix] = ( avgDiffEntries[ix] < 0 ? avgDiffEntries[ix]*-1.: avgDiffEntries[ix] );
  }

  double avgDiff = computeAverage(avgDiffEntries, nx, ny);
  printf("average: %e\n", avg);
  printf("average absolute difference: %e\n", avgDiff);

  // Clean up:
  free(avgDiffEntries);
  free(nextTimeEntries);
  free(ivEntries);
  free(opencl_program_src);

  clReleaseMemObject(new_buffer);
  clReleaseMemObject(old_buffer);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);
  return 0;
}

double computeNextTemp(const double* hij, const double* hijW, const double* hijE, const double* hijS, const double* hijN, const double* c)
{
  return (*hij) + (*c)*( ((*hijW) + (*hijE) + (*hijS) + (*hijN))/4 - (*hij));
}

double computeAverage(const double* tempArray, int nx, int ny)
{
  double sum=0.;
  for ( int ix = 0; ix < nx*ny; ++ix ) {
    sum += tempArray[ix];
  }
  return sum/(nx*ny);
}
