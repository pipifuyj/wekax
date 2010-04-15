function cdset(dsname, N, dim, k)
% CDSET         Create a Dataset having moVMF data and stores as CCS
% 
% Description
% 
% CDSET(DSNAME, N, DIM, K) takes as input:
% 
%  DSNAME : Filename prefix for dataset
%  N      : Num of points to create in set
%  DIM    : Dimensionality of each point
%  K      : Num of clusters to create
%  
%  It produces the following output files:
%  DSNAME_DIM, DSNAME_COL_CCS, DSNAME_ROW_CCS, DSNAME_TFN_NZ
%  DSNAME_LABEL, DSNAME_SOFT_LABEL, DSNAME_PRIORS, DSNAME_CENTERS,
%  DSNAME_KAPPAS
%  
%  The above files are in CCS format and can be used by spkmeans
%  family of programs. The DSNAME_LABEL file has the truelabels.
%  
%  Also see:
%  EMSAMP, MIXINIT, VSAMP

mix = mixinit(dim, k);			% Initialize Data struct


% Now to initialize the mean vectors with random unit vectors.
% Lets init the kappas also with some values (heuristically based
% on dimension)

for i=1:k
  mix.centers(i,:)=unitrand(dim);
  mix.kappas(i) = dim*dim;
  if (dim > 10)
    mix.kappas(i) = mix.kappas(i)*rand;
  end
  if (dim > 20) 
    mix.kappas(i) = max(dim*rand,200);
  end
end

dfile=strcat(dsname, '_centers');
fid = fopen(dfile, 'w');
for i=1:k
  fprintf(fid, '%f ', mix.centers(i,:));
  fprintf(fid, '\n');
end
fclose(fid);

dfile=strcat(dsname, '_kappas');
fid = fopen(dfile, 'w');
for i=1:k
  fprintf(fid, '%f\n', mix.kappas(i));
end
fclose(fid);

[A,l] = emsamp(mix, N);			% Sample data

% Now A is the matrix of the sampled data
% A is N x dim. We want to actually write out the CCS for A'
% Because that corresponds with our notion of documents and words.
% l is the vector of true labels.

% From l we can also gain information about the true priors!

%%%% NOTE: FILE I/O Error checking not done right now

% Write out true label file
lfile = strcat(dsname, '_label');
fid = fopen(lfile, 'w');
l=l-1;					% true label adjust
fprintf(fid, '%d\n', l);                % Write out label
fclose(fid);

% ========================================================
% ADDED On 9/17/02
% Computation of soft labels for the dataset
priors = ones(1,k);			% allocate space for priors
for i=0:k-1
  tmp = find(l==i);
  priors(i+1) = size(tmp,1)/N;
end

dfile = strcat(dsname, '_priors');
fid = fopen(dfile, 'w');
for i=1:k
  fprintf(fid, '%f\n', priors(i));
end
fclose(fid);

% Now we compute the soft label for each point!!!
% p(h|x) -> prior(h) * vmf(x,mix.kappas(h), mix.centers(h))
% Then normalize All these to sum to 1 and voila we have the soft
% labels.
%
probs = zeros(k,1);
dfile = strcat(dsname, '_soft_label');
fid = fopen(dfile, 'w');

for i=1:size(A,1)
  sum = 0;
  for h=1:k
    probs(h) = priors(h)*vmf(A(i,:), mix.centers(h,:), mix.kappas(h));
    sum = sum + probs(h);
  end
  for h=1:k
    probs(h) = probs(h)/ sum;
  end
  for j = 1:k
    fprintf(fid, '%d %f ', j-1, probs(j));
  end
  fprintf(fid,'\n');
end
fclose(fid);

% 
% ========================================================
% Now we begin outputting the CCS format
% First write out the dim file
dfile = strcat(dsname,'_dim');
fid = fopen(dfile, 'w');
fprintf(fid, '%d %d %d', dim, N, N*dim);
fclose(fid);

% write the _words file
wordfile = strcat(dsname,'_words');
fid = fopen(wordfile, 'w');
fprintf(fid, '%d\n', dim);
for i=1:dim
  fprintf(fid, 'feature_%d\n', i);
end
fclose(fid);

% Now we write the _ccs files
dfile = strcat(dsname,'_col_ccs');
fid = fopen(dfile, 'w');
for i=1:N
  fprintf(fid, '%d \n', (i-1)*dim);
end
% print an extra bogus line
fprintf(fid, '%d \n', N*dim);
fclose(fid);

% Now we write the _row_ccs file
dfile = strcat(dsname, '_row_ccs');
fid = fopen(dfile, 'w');
for j=1:N
  for i=0:dim-1
    fprintf(fid,'%d\n', i);
  end
end
fclose(fid);

% Now write out the _nz file
dfile = strcat(dsname, '_tfn_nz');
fid = fopen(dfile, 'w');

% We write out rows of A coz they are A'
for j=1:N
  for i=1:dim
    fprintf(fid, '%f\n', A(j,i));
  end
end

fclose(fid);


% Create arff
CCScommand = ['CCS2arff.pl -i ' dsname];
system(CCScommand);

% rename the arff file
MVcommand = ['mv ' dsname '_fromCCS.arff ' num2str(dim) '-' num2str(k) '-' num2str(N) '.arff'];
system(MVcommand);

% remove CCS files
RMcommand = ['rm -rf ' dsname '_*'];
system(RMcommand);



% Now we have created the dataset
