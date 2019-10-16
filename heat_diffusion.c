#include "heat_diffusion.h"

int main(int argc, char* argv[])
{
  int niter=0;
  double c;
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
  double t=0;
  if ( input == NULL ) {
    perror("Error opening file");
    exit(0);
  }
  //read the first line
  fgets( line, sizeof(line), input);
  sscanf(line, "%d %d", &nx, &ny);
  //store initial values (note: row major order)
  double* iventries = (double*) calloc(ny * nx, sizeof(double));
  double** iv = (double**) malloc(sizeof(double*) * nx); //perhaps not needed if we decide to compute the index manually...
  for (size_t ix = 0, jx = 0; ix < nx; ++ix, jx += ny) {
    iv[ix] = iventries + jx;
  }
  //read the rest
  while ( fgets(line, sizeof(line), input) != NULL) {
    sscanf(line, "%d %d %lf", &i, &j, &t);
    iventries[i*nx + j] = t;
  }
  fclose(input);

  /* for (size_t ix = 0; ix < nx; ++ix) { */
  /*   for (size_t jx = 0; jx < ny; ++jx) { */
  /*     printf("%lf \n", iv[ix][jx]); */
  /*   } */
  /* } */
  
  return 0;
}
