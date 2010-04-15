function fd_1st_d = fD1(X, D, A, N, d)

% ---------------------------------------------------------------------------
% the gradient of the dissimilarity constraint function w.r.t. A
%
% for example, let distance by L1 norm:
% f = f(\sum_{ij \in D} \sqrt{(x_i-x_j)A(x_i-x_j)'})
% df/dA_{kl} = f'* d(\sum_{ij \in D} \sqrt{(x_i-x_j)^k*(x_i-x_j)^l})/dA_{kl}
%
% note that d_ij*A*d_ij' = tr(d_ij*A*d_ij') = tr(d_ij'*d_ij*A)
% so, d(d_ij*A*d_ij')/dA = d_ij'*d_ij
%     df/dA = f'(\sum_{ij \in D} \sqrt{tr(d_ij'*d_ij*A)})
%             * 0.5*(\sum_{ij \in D} (1/sqrt{tr(d_ij'*d_ij*A)})*(d_ij'*d_ij))
% ---------------------------------------------------------------------------
           
sum_dist = 0.000001; sum_deri =  zeros(d,d); 

for i = 1:N
  for j= i+1:N     % count each pair once
    if D(i,j) == 1
      d_ij = X(i,:) - X(j,:);
      [dist_ij, deri_d_ij] = distance1(A, d_ij);
      sum_dist = sum_dist +  dist_ij;
      sum_deri = sum_deri + deri_d_ij;
    end  
  end
end
%sum_dist
fd_1st_d = dgF2(sum_dist)*sum_deri;
