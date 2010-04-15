function [W,H] = nmf(V,r,maxiter,obj)
% NMF - Non-negative Matrix Factorization
% Written by Sebastian Seung, 1999
% See http://journalclub.mit.edu/jclub/home, under Computational
% Neuroscience
%
% Factorizes V into WH so that W has rank r.  This snippet calculates
% W and H by multiplicative update.  The number of iterations is
% maxiter.
%
% From Sebastian Seung:
%
% If you encounter problems with overflow/underflow in computing the
% quotient V./(W*H), you may want to use this version of the code.
% If you find any typos, please let me know.
%
% ChangeLog:
%
% 12/6/2002 - (ywwong) Added objective functions 2 and 3.
%
% 11/13/2002 - (ywwong) Added function header and comment.
%

[n m]=size(V);
W=rand(n,r);              % randomly initialize basis
H=rand(r,m);              % randomly initialize encodings
eps=1e-9;                 % set your own tolerance

% Normalize column sums when maximizing
% F = sum_{i,u}[V_{iu}log(WH)_{iu}-(WH)_{iu}].
if obj == 1
  W=W./(ones(n,1)*sum(W));
end

for iter=1:maxiter
  switch obj
   case 1
    % Maximize F = sum_{i,u}[V_{iu}log(WH)_{iu}-(WH)_{iu}].
    H=H.*(W'*((V+eps)./(W*H+eps)));
    W=W.*(((V+eps)./(W*H+eps))*H');
    W=W./(ones(n,1)*sum(W));
   case 2
    % Minimize F = norm(V-WH).
    H=H.*((W'*V+eps)./(W'*W*H+eps));
    W=W.*((V*H'+eps)./(W*H*H'+eps));
   case 3
    % Minimize F = D(V\|WH).
    H=H.*((W'*((V+eps)./(W*H+eps)))./((sum(W))'*ones(1,m)));
    W=W.*((((V+eps)./(W*H+eps))*H')./(ones(n,1)*(sum(H,2))'));
  end
end

