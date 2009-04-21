function av = unroll(A)

s = size(A);
n = s(1); % # of rows
m = s(2); % # of columns
for i = 1:m
  av( ((i-1)*n+1) : (i*n) ) = A(:,i);
end
av=av';
