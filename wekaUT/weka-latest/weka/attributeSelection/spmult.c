/*
  spmult.c
  Written by Daniel Lee, 2000
  See http://journalclub.mit.edu/jclub/home, under Computational
  Neuroscience
  
  c = spmult(a,b,ic,jc)
  
  Performs matrix multiplication c=a*b, but evaluated only at the sparse
  locations (ic,jc)
*/

#include "mex.h"
#include <math.h>

#define C (plhs[0])
#define A (prhs[0])
#define B (prhs[1])
#define IC (prhs[2])
#define JC (prhs[3])

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int m, n, nc, nz;
  int i, j, jcol, ia0, ib0;
  int *cir, *cjc;
  double *apr, *bpr, *ipr, *jpr, *cpr;
  double s;

  if (nrhs != 4)
    mexErrMsgTxt("Four input arguments required.");

  m = mxGetM(A);
  nc = mxGetN(A);
  if (mxGetM(B) != nc)
    mexErrMsgTxt("Inner matrix dimensions must agree.");
  n = mxGetN(B);
  apr = mxGetPr(A);
  bpr = mxGetPr(B);

  nz = mxGetM(IC)*mxGetN(IC);
  if (mxGetM(JC)*mxGetN(JC) != nz)
    mexErrMsgTxt("Vectors must be the same lengths.");
  ipr = mxGetPr(IC);
  jpr = mxGetPr(JC);
  if ((C = mxCreateSparse(m,n,nz,mxREAL)) == NULL)
    mexErrMsgTxt("Could not allocate sparse matrix.");
  cpr = mxGetPr(C);
  cir = mxGetIr(C);
  cjc = mxGetJc(C);

  jcol = 1;
  cjc[0] = 0;
  for (i=0; i<nz; i++) {
    cir[i] = ipr[i]-1;
    if (jpr[i] < jcol)
      mexErrMsgTxt("Column indices need to be sorted.");
    else {
      while (jpr[i] > jcol) {
        cjc[jcol+1] = cjc[jcol];
        jcol++;
      }
      cjc[jcol]++;
    }

    s = 0.0;
    ia0 = ipr[i]-1;
    ib0 = nc*(jpr[i]-1);
    for (j=0; j<nc; j++) {
      s += apr[ia0]*bpr[ib0];
      ia0 += m;
      ib0++;
    }
    cpr[i] = s;
  }

  while (jcol < n) {
    cjc[jcol+1] = cjc[jcol];
    jcol++;
  }

}

