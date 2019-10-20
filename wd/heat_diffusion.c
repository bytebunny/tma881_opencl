#include "helper.h"
#include "mpi.h"

int
main(int argc, char* argv[])
{
  MPI_Init(NULL, NULL);
  int nmb_mpi_proc, mpi_rank;
  MPI_Comm mpiworld = MPI_COMM_WORLD;

  MPI_Comm_size(mpiworld, &nmb_mpi_proc);
  MPI_Comm_rank(mpiworld, &mpi_rank);

  int niter = 0;
  double c = 0.0;
  char* ptr1 = NULL;
  char* ptr2 = NULL;
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
  FILE* input = fopen("diffusion", "r");
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
  double* ivEntries = (double*)calloc(sizeof(double), nx * ny);

  if (mpi_rank == 0) {
    while (fgets(line, sizeof(line), input) != NULL) {
      sscanf(line, "%d %d %lf", &i, &j, &t);
      ivEntries[i * ny + j] = t;
    }
  }
  fclose(input);

  size_t const ix_sub_m = (nx - 1) / nmb_mpi_proc + 1;
  int *lens, *poss;
  lens = (int*)malloc(nmb_mpi_proc * sizeof(int));
  poss = (int*)malloc(nmb_mpi_proc * sizeof(int));

  for (size_t jx = 0, pos = 0; jx < nmb_mpi_proc; ++jx, pos += ix_sub_m) {
    lens[jx] = ix_sub_m < nx - pos ? ix_sub_m : nx - pos;
    poss[jx] = pos;
  }

  // compute the temperatures in next time step
  for (int n = 0; n < niter; ++n) {

    if (mpi_rank == 0) {
      double* nextTimeEntries = (double*)malloc(sizeof(double) * nx * ny);
      for (size_t rk = 0; rk < nmb_mpi_proc; rk++) {
        int addup, addlo;
        addup = (rk == 0 ? 0 : 1);
        addlo = (rk == (nmb_mpi_proc - 1) ? 0 : 1);
        size_t loc_len = lens[rk] + addup + addlo;
        double* loc_heats = (double*)malloc(sizeof(double) * loc_len * ny);
        memcpy(loc_heats,
               ivEntries + (poss[rk] - addup) * ny,
               sizeof(double) * loc_len * ny);
        if (rk == 0) {
          for (int ix = addup; ix < lens[mpi_rank]; ++ix) {
            for (int jx = 0; jx < ny; ++jx) {

              double hij, hijW, hijE, hijS, hijN;
              hij = loc_heats[ix * ny + jx];
              hijW = (jx - 1 >= 0 ? loc_heats[ix * ny + jx - 1] : 0.);
              hijE = (jx + 1 < ny ? loc_heats[ix * ny + jx + 1] : 0.);
              hijS = (ix + 1 < nx ? loc_heats[(ix + 1) * ny + jx] : 0.);
              hijN = (ix - 1 >= 0 ? loc_heats[(ix - 1) * ny + jx] : 0.);
              nextTimeEntries[ix * ny + jx] =
                hij + c * ((hijW + hijE + hijS + hijN) / 4 - hij);
            }
          }
        } else {
          MPI_Ssend(loc_heats, loc_len * ny, MPI_DOUBLE, rk, 0, mpiworld);
        }
        free(loc_heats);
      }

      if (nmb_mpi_proc > 1) {
        for (size_t rk = 1; rk < nmb_mpi_proc; rk++) {
          double* loc_heats_recv =
            (double*)malloc(sizeof(double) * lens[rk] * ny);
          MPI_Recv(
            loc_heats_recv, lens[rk] * ny, MPI_DOUBLE, rk, 0, mpiworld, NULL);

          for (size_t jx = 0; jx < lens[rk]; jx++) {
            for (size_t kx = 0; kx < ny; kx++) {
              nextTimeEntries[(poss[rk] + jx) * ny + kx] =
                loc_heats_recv[jx * ny + kx];
            }
          }
          free(loc_heats_recv);
        }
      }
      memcpy(ivEntries, nextTimeEntries, nx * ny * sizeof(double));

      free(nextTimeEntries);

    } else {
      int addup, addlo;
      addup = (mpi_rank == 0 ? 0 : 1);
      addlo = (mpi_rank == (nmb_mpi_proc - 1) ? 0 : 1);
      size_t loc_len = lens[mpi_rank] + addup + addlo;
      double* loc_heats_recv = (double*)malloc(sizeof(double) * loc_len * ny);
      MPI_Status status;
      MPI_Recv(
        loc_heats_recv, loc_len * ny, MPI_DOUBLE, 0, 0, mpiworld, &status);

      double* next_loc_heats =
        (double*)malloc(sizeof(double) * lens[mpi_rank] * ny);

      for (int ix = 0; ix < lens[mpi_rank]; ++ix) {
        for (int jx = 0; jx < ny; ++jx) {
          double hij, hijW, hijE, hijS, hijN;
          size_t loc_ix = ix + addup;
          hij = loc_heats_recv[loc_ix * ny + jx];
          hijW = (jx - 1 >= 0 ? loc_heats_recv[loc_ix * ny + jx - 1] : 0.);
          hijE = (jx + 1 < ny ? loc_heats_recv[loc_ix * ny + jx + 1] : 0.);
          hijS =
            (loc_ix + 1 < nx ? loc_heats_recv[(loc_ix + 1) * ny + jx] : 0.);
          hijN =
            (loc_ix - 1 >= 0 ? loc_heats_recv[(loc_ix - 1) * ny + jx] : 0.);
          next_loc_heats[ix * ny + jx] =
            hij + c * ((hijW + hijE + hijS + hijN) / 4 - hij);
        }
      }
      //   printf("rank %d compute\n", mpi_rank);
      free(loc_heats_recv);
      MPI_Ssend(
        next_loc_heats, lens[mpi_rank] * ny, MPI_DOUBLE, 0, 0, mpiworld);
      free(next_loc_heats);
    }
  } // end of iteration

  if (mpi_rank == 0) {
    printf("updated\n");
    for (size_t ii = 0; ii < nx; ii++) {
      printf("%f %f %f\n",
             ivEntries[ii * ny],
             ivEntries[ii * ny + 1],
             ivEntries[ii * ny + 2]);
    }

    double avg = computeAverage(ivEntries, nx, ny);

    double* avgDiffEntries = (double*)calloc(sizeof(double), nx * ny);

    memcpy(avgDiffEntries, ivEntries, nx * ny * sizeof(double));

    for (int ix = 0; ix < nx * ny; ++ix) {
      avgDiffEntries[ix] -= avg;
      avgDiffEntries[ix] = (avgDiffEntries[ix] < 0 ? -1.0 * avgDiffEntries[ix]
                                                   : avgDiffEntries[ix]);
    }

    double avgDiff = computeAverage(avgDiffEntries, nx, ny);
    printf("average: %e\n", avg);
    printf("average absolute difference: %e\n", avgDiff);

    free(avgDiffEntries);
  }

  free(ivEntries);
  free(poss);
  free(lens);
  MPI_Finalize();
  return 0;
}

double
computeAverage(const double* tempArray, int nx, int ny)
{
  double sum = 0.;
  for (int ix = 0; ix < nx * ny; ++ix) {
    sum += tempArray[ix];
  }
  return sum / (nx * ny);
}
