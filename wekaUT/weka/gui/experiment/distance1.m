function [dist_ij, deri_d_ij] = distance1(A, d_ij)
% distance and derivative of distance using distance1: distance(d) = L1
fudge = 0.000001;  % regularizes derivates a little

      M_ij = d_ij'*d_ij;
      dist_ij = sqrt(trace(M_ij*A));

      % derivative of dist_ij w.r.t. A
      deri_d_ij = 0.5*M_ij/(dist_ij+fudge); 
