function [ B, A, newData ] = RCA(TrainingSet,chunkletAssignments,priorFlag)

if(~exist('priorFlag'))
    priorFlag=1;
end

TrainingSet=TrainingSet';	% tomer conversion
S= max(chunkletAssignments);
ChunkletDataSet= [];
ChunkletSizes= [];
for i=1:S
    inds= find(chunkletAssignments == i);
    ChunkletSizes(i)= length(inds);
    ChunkletDataSet= [ChunkletDataSet TrainingSet(:,inds)]
end

[CenteredChunklets]=centerChunklets(ChunkletDataSet,ChunkletSizes);

%compute the svd of the normalized chunklets
Cov= CenteredChunklets*CenteredChunklets';
[dim N]= size(CenteredChunklets);

%check for ill ranked RCA covmat in case of not enough chunklets: if
%so smooth matrix:
alpha= N/50;
if(priorFlag)&(rank(Cov) < dim)
    Cov= (Cov + alpha.*eye(dim))/(sum(ChunkletSizes)+alpha);
else  
    Cov= Cov./N;
end

[U Sig V]= svd(Cov);

% whiten the data set using RCA Transformation:

A=(Sig^-0.5)*U';
newData = A*TrainingSet;

% the mahalanovis matrix
B=inv(Cov);


%%%%%%%%%%% end of RCA function
