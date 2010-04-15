function const = calc_const(p,k)
% CALC_CONST  calculates the log of the normalizing constant
% based on the p and kappa.

p = p/2 - 1;
const = (k*0.5/pi)^p;
temp = 2*pi*besseli(p,k);
const = log(const) - log(temp);