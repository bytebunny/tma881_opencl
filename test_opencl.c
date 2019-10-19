#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>

int main(){
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
  FILE* programHandle;
  size_t programSize;
  programHandle = fopen("dot_prod_mul.cl", "r");
  fseek(programHandle, 0, SEEK_END);
  programSize = ftell(programHandle);
  rewind(programHandle);
  
  // read kernel source into buffer
  char * opencl_program_src = (char*) malloc(programSize + 1);
  opencl_program_src[programSize] = '\0';
  fread(opencl_program_src, sizeof(char), programSize, programHandle);
  fclose(programHandle);

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
    clGetProgramBuildInfo( program, device_id[0], CL_PROGRAM_BUILD_LOG,
                           0, NULL, &log_size); // get log_size.

    char * log = calloc(log_size, sizeof(char));
    if (log == NULL) {
      printf("could not allocate memory\n");
      return(1);
    }
    clGetProgramBuildInfo( program, device_id[0], CL_PROGRAM_BUILD_LOG,
                           log_size, log, NULL); // get log info.
    printf( "%s\n", log );
    
    free(log);
    return(1);
  }

  /* // Create kernel (contained in program): */
  cl_kernel kernel;
  kernel = clCreateKernel( program, (const char *)"dot_prod_mul", &error_code );
  if (error_code != CL_SUCCESS) {
    printf("cannot create kernel\n");
    return(1);
  }

  //Create memory objects (within context):
  const size_t ix_m = 10e7;
  cl_mem input_buffer_a, input_buffer_b, output_buffer_c;
  input_buffer_a  = clCreateBuffer( context, CL_MEM_READ_ONLY,
                                    sizeof(cl_float) * ix_m, NULL, &error_code );
  if (error_code != CL_SUCCESS) {
    printf("cannot create buffer for a\n");
    return(1);
  }
  input_buffer_b  = clCreateBuffer( context, CL_MEM_READ_ONLY,
                                    sizeof(cl_float) * ix_m, NULL, &error_code );
  if (error_code != CL_SUCCESS) {
    printf("cannot create buffer for b\n");
    return(1);
  }
  output_buffer_c = clCreateBuffer( context, CL_MEM_WRITE_ONLY,
                                    sizeof(cl_float) * ix_m, NULL, &error_code );
  if (error_code != CL_SUCCESS) {
    printf("cannot create buffer for c\n");
    return(1);
  }

  // Copy memory objects from host to buffer object:
  cl_float * a = (cl_float *) malloc(ix_m*sizeof(cl_float));
  cl_float * b = (cl_float *) malloc(ix_m*sizeof(cl_float));
  for (size_t ix = 0; ix < ix_m; ++ix) {
    a[ix] = ix;
    b[ix] = ix;
  }

  error_code = clEnqueueWriteBuffer( command_queue, input_buffer_a, CL_TRUE,
                                     0, // offset in bytes in the buffer object to write to.
                                     ix_m*sizeof(cl_float), (const void *)a,
                                     0, NULL, NULL ); // events that need to to complete before this particular command can be executed.
  if (error_code != CL_SUCCESS) {
    printf("cannot enqueue writing to buffer for a\n");
    return(1);
  }

  error_code = clEnqueueWriteBuffer( command_queue, input_buffer_b, CL_TRUE,
                                     0, // offset in bytes in the buffer object to write to.
                                     ix_m*sizeof(cl_float), (const void *)b,
                                     0, NULL, NULL ); // events that need to to complete before this particular command can be executed.
  if (error_code != CL_SUCCESS) {
    printf("cannot enqueue writing to buffer for b\n");
    return(1);
  }

  // Set kernel arguments:
  error_code = clSetKernelArg( kernel, (cl_uint)0,
                               sizeof(cl_mem), (const void *)&input_buffer_a );
  if (error_code != CL_SUCCESS) {
    printf("cannot set kernel arguments for a\n");
    return(1);
  }
  error_code = clSetKernelArg( kernel, (cl_uint)1,
                               sizeof(cl_mem), (const void *)&input_buffer_b );
  if (error_code != CL_SUCCESS) {
    printf("cannot set kernel arguments for b\n");
    return(1);
  }
  error_code = clSetKernelArg( kernel, (cl_uint)2,
                               sizeof(cl_mem), (const void *)&output_buffer_c );
  if (error_code != CL_SUCCESS) {
    printf("cannot set kernel arguments for c\n");
    return(1);
  }
  
  // Put kernel into queue -> execution:
  const size_t global = ix_m;
  clEnqueueNDRangeKernel( command_queue, kernel,
                          (cl_uint)1, // number
                          NULL, (const size_t *)&global, NULL, 0, NULL, NULL );



  // Copy memory objects with results from device to host
  cl_float * c = (cl_float *) malloc(ix_m*sizeof(float));
  clEnqueueReadBuffer( command_queue, output_buffer_c, CL_TRUE,
                       0, ix_m*sizeof(float), (void *)c, 0, NULL, NULL );

  clFinish(command_queue);
  
  printf("Square of %d is %f\n", 100, c[100]);

  // Clean up:
  free(a);
  free(b);
  free(c);
  free(opencl_program_src);

  clReleaseMemObject(input_buffer_a);
  clReleaseMemObject(input_buffer_b);
  clReleaseMemObject(output_buffer_c);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);
  return(0);
}
