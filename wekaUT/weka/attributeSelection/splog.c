/*
  splog.c
  Written by Daniel Lee, 2000
  See http://journalclub.mit.edu/jclub/home, under Computational
  Neuroscience
  
  y = splog(x)
  
  Performs matrix element log: y=log(x), but evaluated only at the
  sparse locations.
*/

#include "mex.h"
#include <string.h>
#include <math.h>

#define Y (plhs[0])
#define X (prhs[0])

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int m, n, nzmax, nnz;
  int i;
  double *xpr, *ypr;

  if (nrhs != 1)
    mexErrMsgTxt("One input arguments required.");
  if (!mxIsSparse(X))
    mexErrMsgTxt("Input argument must be sparse.");

  m = mxGetM(X);
  n = mxGetN(X);
  nzmax = mxGetNzmax(X);
  nnz = *(mxGetJc(X)+n);

  xpr = mxGetPr(X);

  if ((Y = mxCreateSparse(m,n,nzmax,mxREAL)) == NULL)
    mexErrMsgTxt("Could not allocate sparse matrix.");
  ypr = mxGetPr(Y);

  memcpy(mxGetIr(Y), mxGetIr(X), nnz*sizeof(int));
  memcpy(mxGetJc(Y), mxGetJc(X), (n+1)*sizeof(int));

  for (i=0; i<nnz; i++)
    ypr[i] = log(xpr[i]);

}

