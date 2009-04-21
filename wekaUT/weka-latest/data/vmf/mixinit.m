function mixture = mixinit(dim, num_clus)
% MIXINIT	Allocates matrices for mixture data structure.
% 
% MIXTURE=MIXINIT(DIM, NUM_CLUS)

% Set the dimensionality ... corr to 'd'
mixture.dim = dim;

% Equal priors for all clusters.
mixture.priors = ones(1,num_clus)/num_clus;

% Record no. of clusters we are dealing with.
mixture.num_clus = num_clus;

% Initialize centroids to be null.
mixture.centers = zeros(num_clus,dim);
mixture.centers(:,dim)=1;
% The covars(:,:,i) corresponds to \Sigma_i for the mixture.
mixture.kappas = 10*ones(1,num_clus);