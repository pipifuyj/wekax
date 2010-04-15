/*
  spdotdiv.c
  Written by Daniel Lee, 2000
  See http://journalclub.mit.edu/jclub/home, under Computational
  Neuroscience
  
  c = spdotdiv(a,b)
  
  Performs matrix element division c=a./b, but evaluated only at the
  sparse locations. (a and b must have same sparcity structure).
*/

#include "mex.h"
#include <string.h>
#include <math.h>

#define C (plhs[0])
#define A (prhs[0])
#define B (prhs[1])

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int m, n, nzmax, nnz;
  int i;
  double *apr, *bpr, *cpr;

  if (nrhs != 2)
    mexErrMsgTxt("Two input arguments required.");
  if (!mxIsSparse(A) || !mxIsSparse(B))
    mexErrMsgTxt("Input arguments must be sparse.");

  m = mxGetM(A);
  n = mxGetN(A);
  nzmax = mxGetNzmax(A);
  nnz = *(mxGetJc(A)+n);

  if ((mxGetM(B) != m) || (mxGetN(B) != n) || (mxGetNzmax(B) != nzmax))
    mexErrMsgTxt("Input matrices must have same sparcity structure.");

  apr = mxGetPr(A);
  bpr = mxGetPr(B);

  if ((C = mxCreateSparse(m,n,nzmax,mxREAL)) == NULL)
    mexErrMsgTxt("Could not allocate sparse matrix.");
  cpr = mxGetPr(C);

  memcpy(mxGetIr(C), mxGetIr(A), nnz*sizeof(int));
  memcpy(mxGetJc(C), mxGetJc(A), (n+1)*sizeof(int));

  for (i=0; i<nnz; i++)
    cpr[i] = apr[i]/bpr[i];

}


