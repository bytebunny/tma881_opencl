#include "helper.h"
#include "mpi.h"

int main(int argc, char *argv[]) {
  MPI_Init(NULL, NULL);
  double strtim = MPI_Wtime();
  int nmb_mpi_proc, mpi_rank;
  MPI_Comm mpiworld = MPI_COMM_WORLD;

  MPI_Comm_size(mpiworld, &nmb_mpi_proc);
  MPI_Comm_rank(mpiworld, &mpi_rank);

  if (mpi_rank == 0) {
    printf("use %d nodes\n", nmb_mpi_proc);
  }

  int niter = 0;
  double c = 0.0;
  char *ptr1 = NULL;
  char *ptr2 = NULL;
  if (argc == 3) {
    for (size_t ix = 1; ix < argc; ++ix) {
      ptr1 = strchr(argv[ix], 'n');
      ptr2 = strchr(argv[ix], 'd');
      if (ptr1) {
        niter = strtol(++ptr1, NULL, 10);
      } else if (ptr2) {
        c = atof(++ptr2);
      }
    }
  } else {
    printf("Invalid number of arguments. Correct syntax is: heat_diffusion "
           "-n#numberOfTimeSteps4 -d#diffusionConstant\n");
    return 1;
  }

  int nx = 0, ny = 0;
  char line[80];
  FILE *input = fopen("diffusion", "r");
  int i = 0, j = 0;
  double t = 0.;
  if (input == NULL) {
    perror("Error opening file");
    return 2;
  }

  // read the first line
  fgets(line, sizeof(line), input);
  sscanf(line, "%d %d", &nx, &ny);
  // store initial values (note: row major order)
  double *ivEntries = (double *)malloc(sizeof(double) * nx * ny);
  double *nextTimeEntries = (double *)malloc(sizeof(double) * nx * ny);

  while (fgets(line, sizeof(line), input) != NULL) {
    sscanf(line, "%d %d %lf", &i, &j, &t);
    ivEntries[i * ny + j] = t;
  }

  fclose(input);

  size_t const ix_sub_m = (nx - 1) / nmb_mpi_proc + 1;
  int *lens = (int *)malloc(nmb_mpi_proc * sizeof(int));
  int *poss = (int *)malloc(nmb_mpi_proc * sizeof(int));

  for (size_t jx = 0, pos = 0; jx < nmb_mpi_proc; ++jx, pos += ix_sub_m) {
    lens[jx] = ix_sub_m < nx - pos ? ix_sub_m : nx - pos;
    poss[jx] = pos;
  }

  double sum;
  // compute the temperatures in next time step
  for (int n = 0; n < niter; ++n) {
    sum = 0.0;
    // if (n > 0) {
    // MPI_Bcast(ivEntries, nx * ny, MPI_DOUBLE, 0, mpiworld);
    // }

    for (int ix = 0; ix < lens[mpi_rank]; ++ix) {
      int rind = poss[mpi_rank] + ix;
      for (int jx = 0; jx < ny; ++jx) {
        double hij, hijW, hijE, hijS, hijN;
        hij = ivEntries[rind * ny + jx];
        hijW = (jx - 1 >= 0 ? ivEntries[rind * ny + jx - 1] : 0.);
        hijE = (jx + 1 < ny ? ivEntries[rind * ny + jx + 1] : 0.);
        hijS = (rind + 1 < nx ? ivEntries[(rind + 1) * ny + jx] : 0.);
        hijN = (rind - 1 >= 0 ? ivEntries[(rind - 1) * ny + jx] : 0.);
        nextTimeEntries[rind * ny + jx] =
            hij + c * ((hijW + hijE + hijS + hijN) / 4 - hij);
        sum += nextTimeEntries[rind * ny + jx];
      }
    }
    memcpy(ivEntries + poss[mpi_rank] * ny,
           nextTimeEntries + poss[mpi_rank] * ny,
           lens[mpi_rank] * ny * sizeof(double));

    if (nmb_mpi_proc > 1) {
      if (mpi_rank == 0) {
        // send the curren rank last to the right rank
        double *sendind =
            nextTimeEntries + (poss[mpi_rank] + lens[mpi_rank] - 1) * ny;
        MPI_Ssend(sendind, ny, MPI_DOUBLE, 1, 0, mpiworld);
      } else {
        // receive left rank last to the curren rank top - 1
        double *recvind = ivEntries + (poss[mpi_rank] - 1) * ny;
        MPI_Recv(recvind, ny, MPI_DOUBLE, mpi_rank - 1, 0, mpiworld, NULL);
        // send the curren rank last to the right rank
        if (mpi_rank + 1 < nmb_mpi_proc) {
          double *sendind =
              nextTimeEntries + (poss[mpi_rank] + lens[mpi_rank] - 1) * ny;
          MPI_Ssend(sendind, ny, MPI_DOUBLE, mpi_rank + 1, 0, mpiworld);
        }
      }
      // MPI_Barrier(mpiworld);

      if (mpi_rank == nmb_mpi_proc - 1) {
        // send the curren rank last to the right rank
        double *sendind = nextTimeEntries + poss[mpi_rank] * ny;
        MPI_Ssend(sendind, ny, MPI_DOUBLE, mpi_rank - 1, 0, mpiworld);
      } else {
        // receive right rank last to the curren rank top - 1
        double *recvind = ivEntries + (poss[mpi_rank] + lens[mpi_rank]) * ny;
        MPI_Recv(recvind, ny, MPI_DOUBLE, mpi_rank + 1, 0, mpiworld, NULL);
        if (mpi_rank > 0) {
          // send the curren rank last to the right rank
          double *sendind = nextTimeEntries + poss[mpi_rank] * ny;
          MPI_Ssend(sendind, ny, MPI_DOUBLE, mpi_rank - 1, 0, mpiworld);
        }
      }
      /*
            printf("iter %d rank %d\n", n, mpi_rank);
            for (size_t ii = 0; ii < nx; ii++) {
              printf("%f %f %f\n", ivEntries[ii * ny], ivEntries[ii * ny + 1],
                     ivEntries[ii * ny + 2]);
            }
       */
      //MPI_Barrier(mpiworld);
    }
  } // end of iteration

  double totalSum = 0.0;
  MPI_Reduce(&sum, &totalSum, 1, MPI_DOUBLE, MPI_SUM, 0, mpiworld);
  double avg = totalSum / (double)(nx * ny);
  MPI_Bcast(&avg, 1, MPI_DOUBLE, 0, mpiworld);

  double abssum = 0.0;
  for (int ix = 0; ix < lens[mpi_rank]; ++ix) {
    int rind = poss[mpi_rank] + ix;
    for (int jx = 0; jx < ny; ++jx) {
      double absdelta = nextTimeEntries[rind * ny + jx] - avg;
      absdelta = (absdelta < 0.0 ? -absdelta : absdelta);
      abssum += absdelta;
    }
  }

  double totalAbssum = 0.0;
  MPI_Reduce(&abssum, &totalAbssum, 1, MPI_DOUBLE, MPI_SUM, 0, mpiworld);
  double absavg = totalAbssum / (double)(nx * ny);


  free(nextTimeEntries);
  free(ivEntries);
  free(poss);
  free(lens);
  if (mpi_rank == 0) {

    printf("average: %e\n", avg);
    printf("average absolute difference: %e\n", absavg);

    double endtim = MPI_Wtime();
    double tiktork = endtim - strtim;
    printf("run time %f\n", tiktork);
  }
  MPI_Finalize();
  return 0;
}
