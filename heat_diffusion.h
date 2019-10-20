#ifndef HEATDIFF_H
#define HEATDIFF_H

double computeNextTemp(const double* hij, const double* hijW, const double* hijE, const double* hijS, const double* hijN, const double* c);
double computeAverage(const double* tempArray, int nx, int ny);

#endif
