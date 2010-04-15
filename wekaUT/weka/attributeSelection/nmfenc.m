function h = nmfenc(v,W,maxiter,obj)
% NMF - Non-negative Matrix Factorization
% Written by Sebastian Seung, 1999
% See http://journalclub.mit.edu/jclub/home, under Computational
% Neuroscience
%
% Given a basis W, find an encoding h for a vector v.  This snippet
% calculates h by multiplicative update.  The number of iterations
% is maxiter.
%
% ChangeLog:
%
% 12/6/2002 - (ywwong) Added objective functions 2 and 3.
%
% 11/13/2002 - (ywwong) Adapted from nmf.m.
%

r=size(W,2);
h=rand(r,1);              % randomly initialize encoding
eps=1e-9;                 % set your own tolerance

for iter=1:maxiter
  switch obj
   case 1
    % Maximize F = sum_{i}[v_{i}log(Wh)_{i}-(Wh)_{i}].
    h=h.*(W'*((v+eps)./(W*h+eps)));
   case 2
    % Minimize F = norm(v-Wh).
    h=h.*((W'*v+eps)./(W'*W*h+eps));
   case 3
    % Minimize F = D(v\|Wh).
    h=h.*((W'*((v+eps)./(W*h+eps)))./(sum(W))');
  end
end
