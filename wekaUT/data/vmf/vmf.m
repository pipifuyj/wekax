function pdf = vmf(x, mu, kappa)
% VMF         Returns the VMF Prob. density at point x
%
% PDF = VMF(X,MU,KAPPA)
%
% Computes and returns the von-Mises Fisher probability density at
% a given piont X, where the density has params MU and KAPPA
% It assumes that MU, X are input as row vectors.
% 
% See: VSAMP, EMSAMP, CDSET

tmp = calc_const(size(mu,2), kappa);
t2 = dot(mu,x);
pdf = exp(tmp) * exp(kappa* t2);
