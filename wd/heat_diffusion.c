#include "helper.h"
#include "mpi.h"

int
main(int argc, char* argv[])
{
  MPI_Init(NULL, NULL);
  // double strtim = MPI_Wtime();
  int nmb_mpi_proc, mpi_rank;
  MPI_Comm mpiworld = MPI_COMM_WORLD;

  MPI_Comm_size(mpiworld, &nmb_mpi_proc);
  MPI_Comm_rank(mpiworld, &mpi_rank);

  if (mpi_rank == 0) {
    // printf("use %d nodes\n", nmb_mpi_proc);
  }

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
  double* ivEntries = (double*)calloc(sizeof(double), (nx + 2) * (ny + 2));
  double* nextTimeEntries =
    (double*)calloc(sizeof(double), (nx + 2) * (ny + 2));

  while (fgets(line, sizeof(line), input) != NULL) {
    sscanf(line, "%d %d %lf", &i, &j, &t);
    ivEntries[(i + 1) * (ny + 2) + j + 1] = t;
  }

  fclose(input);

  size_t const ix_sub_m = (nx - 1) / nmb_mpi_proc + 1;
  // int *lens = (int *)malloc(nmb_mpi_proc * sizeof(int));
  // int *poss = (int *)malloc(nmb_mpi_proc * sizeof(int));
  int lens[nmb_mpi_proc];
  int poss[nmb_mpi_proc];

  for (size_t jx = 0, pos = 0; jx < nmb_mpi_proc; ++jx, pos += ix_sub_m) {
    lens[jx] = ix_sub_m < nx - pos ? ix_sub_m : nx - pos;
    poss[jx] = pos + 1;
  }

  double sum = 0.0;
  // double tmpdata[nmb_mpi_proc * (ny + 2) * 2];
  // double sendind[2 * (ny + 2)];
  // compute the temperatures in next time step
  for (int n = 0; n < niter; ++n) {
    for (int ix = 0; ix < lens[mpi_rank]; ++ix) {
      int rind = poss[mpi_rank] + ix;
      for (int jx = 1; jx < ny + 1; ++jx) {
        double hij = ivEntries[rind * (ny + 2) + jx];
        double hijW = ivEntries[rind * (ny + 2) + jx - 1];
        double hijE = ivEntries[rind * (ny + 2) + jx + 1];
        double hijS = ivEntries[(rind + 1) * (ny + 2) + jx];
        double hijN = ivEntries[(rind - 1) * (ny + 2) + jx];
        nextTimeEntries[rind * (ny + 2) + jx] =
          hij + c * ((hijW + hijE + hijS + hijN) / 4 - hij);
        if (n + 1 == niter) {
          sum += nextTimeEntries[rind * (ny + 2) + jx];
        }
      }
    }
    double* swp = nextTimeEntries;
    nextTimeEntries = ivEntries;
    ivEntries = swp;
    //    memcpy(ivEntries + poss[mpi_rank] * (ny + 2),
    //           nextTimeEntries + poss[mpi_rank] * (ny + 2),
    //           lens[mpi_rank] * (ny + 2) * sizeof(double));

    if (nmb_mpi_proc > 1) {

      /*
       * ─── BRADCAST
       * ───────────────────────────────────────────────────────────────────
       */
      /*
            if (mpi_rank > 0) {
              memcpy(sendind,
                     nextTimeEntries + (poss[mpi_rank]) * (ny + 2),
                     sizeof(double) * (ny + 2));
              memcpy(sendind + (ny + 2),
                     nextTimeEntries +
                       (poss[mpi_rank] + lens[mpi_rank] - 1) * (ny + 2),
                     sizeof(double) * (ny + 2));
              MPI_Send(sendind, (ny + 2) * 2, MPI_DOUBLE, 0, 0, mpiworld);
            } else {
              memcpy(tmpdata, nextTimeEntries + ny + 2, sizeof(double) * (ny +
         2)); memcpy(tmpdata + ny + 2, nextTimeEntries + (ny + 2) * lens[0],
                     sizeof(double) * (ny + 2));

              for (size_t rk = 1; rk < nmb_mpi_proc; rk++) {
                MPI_Recv(tmpdata + (rk) * (ny + 2) * 2,
                         (ny + 2) * 2,
                         MPI_DOUBLE,
                         rk,
                         0,
                         mpiworld,
                         NULL);
              }
            }

            MPI_Bcast(tmpdata, nmb_mpi_proc * (ny + 2) * 2, MPI_DOUBLE, 0,
         mpiworld);
            // printf("rank %d\n", mpi_rank);
            // for (size_t ii = 1; ii < nx + 1; ii++) {
            //   for (size_t ij = 1; ij < ny + 1; ij++) {
            //     printf("%f\t", ivEntries[ii * (ny + 2) + ij]);
            //   }
            //   printf("\b\n");
            // }

            if (mpi_rank > 0) {
              memcpy(ivEntries + (poss[mpi_rank] - 1) * (ny + 2),
                     tmpdata + (mpi_rank - 1) * (ny + 2) * 2 + ny + 2,
                     sizeof(double) * (ny + 2));
            }
            if (mpi_rank + 1 < nmb_mpi_proc) {
              memcpy(ivEntries + (poss[mpi_rank] + lens[mpi_rank]) * (ny + 2),
                     tmpdata + (mpi_rank + 1) * (ny + 2) * 2,
                     sizeof(double) * (ny + 2));
            }
      */

      /*
       * ─── RING
       * ───────────────────────────────────────────────────────────────────────
       */

      if (nmb_mpi_proc > 1) {
        // messaging from rank 0 to rank n
        if (mpi_rank == 0) {

          // send the curren rank last line to the right rank
          double* sendind =
            ivEntries + (poss[mpi_rank] + lens[mpi_rank] - 1) * (ny + 2);
          MPI_Ssend(sendind, ny + 2, MPI_DOUBLE, 1, 0, mpiworld);
          // receive right rank top line to the curren rank btm - 1
          double* recvind =
            ivEntries + (poss[mpi_rank] + lens[mpi_rank]) * (ny + 2);
          MPI_Recv(recvind, ny + 2, MPI_DOUBLE, 1, 0, mpiworld, NULL);
        } else if (mpi_rank == nmb_mpi_proc - 1) {

          // receive left rank last line to the curren rank top - 1
          double* recvind = ivEntries + (poss[mpi_rank] - 1) * (ny + 2);
          MPI_Recv(
            recvind, ny + 2, MPI_DOUBLE, mpi_rank - 1, 0, mpiworld, NULL);
          // send the curren rank top line to the left rank
          double* sendind = ivEntries + (poss[mpi_rank]) * (ny + 2);
          MPI_Ssend(sendind, ny + 2, MPI_DOUBLE, mpi_rank - 1, 0, mpiworld);
        } else {

          // receive left rank last line to the curren rank top - 1
          double* recvindl2r = ivEntries + (poss[mpi_rank] - 1) * (ny + 2);
          MPI_Recv(
            recvindl2r, ny + 2, MPI_DOUBLE, mpi_rank - 1, 0, mpiworld, NULL);
          // send the curren rank last line to the right rank
          double* sendindl2r =
            ivEntries + (poss[mpi_rank] + lens[mpi_rank] - 1) * (ny + 2);
          MPI_Ssend(sendindl2r, ny + 2, MPI_DOUBLE, mpi_rank + 1, 0, mpiworld);
          /* • • • • • */
          // receive right rank top line to the curren rank btm - 1
          double* recvindr2l =
            ivEntries + (poss[mpi_rank] + lens[mpi_rank]) * (ny + 2);
          MPI_Recv(
            recvindr2l, ny + 2, MPI_DOUBLE, mpi_rank + 1, 0, mpiworld, NULL);
          // send the curren rank top line to the right rank
          double* sendindr2l = ivEntries + (poss[mpi_rank]) * (ny + 2);
          MPI_Ssend(sendindr2l, ny + 2, MPI_DOUBLE, mpi_rank - 1, 0, mpiworld);
        }
      }
    }
  } // end of iteration

  double redSum = 0.0;
  MPI_Reduce(&sum, &redSum, 1, MPI_DOUBLE, MPI_SUM, 0, mpiworld);
  double avg = redSum / (double)(nx * ny);
  /*
  double abssum = 0.0;
  for (int ix = 1; ix < nx + 1; ++ix) {
    for (int jx = 1; jx < ny + 1; ++jx) {
      double absdelta = ivEntries[ix * (ny + 2) + jx] - avg;
      absdelta = (absdelta < 0.0 ? -absdelta : absdelta);
      abssum += absdelta;
    }
  }
 */
  MPI_Bcast(&avg, 1, MPI_DOUBLE, 0, mpiworld);
  double abssum = 0.0;
  for (int ix = 0; ix < lens[mpi_rank]; ++ix) {
    int rind = poss[mpi_rank] + ix;
    for (int jx = 1; jx < ny + 1; ++jx) {
      double absdelta = ivEntries[rind * (ny + 2) + jx] - avg;
      absdelta = (absdelta < 0.0 ? -absdelta : absdelta);
      abssum += absdelta;
    }
  }

  /*
  printf("rank %d\n", mpi_rank);
  for (size_t ii = 1; ii < nx + 1; ii++) {
    for (size_t ij = 1; ij < ny + 1; ij++) {
      printf("%f\t", ivEntries[ii * (ny + 2) + ij]);
    }
    printf("\b\n");
  }
   */

  // double absavg = abssum / (double)(nx * ny);

  double redAbsSum = 0.0;
  MPI_Reduce(&abssum, &redAbsSum, 1, MPI_DOUBLE, MPI_SUM, 0, mpiworld);
  double absavg = redAbsSum / (double)(nx * ny);

  free(nextTimeEntries);
  free(ivEntries);
  // free(poss);
  // free(lens);

  if (mpi_rank == 0) {

    printf("average: %e\n", avg);
    printf("average absolute difference: %e\n", absavg);

    // double endtim = MPI_Wtime();
    // double tiktork = endtim - strtim;
    // printf("run time %f\n", tiktork);
  }

  MPI_Finalize();
  return 0;
}
