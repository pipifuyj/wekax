% This function computes the centroid of each chunklet and substracts it from all
% chunklet members.
% ChunkletSizes is a vector containing the size of each chunklet in the data set.
% chunklets are assumed to be ordered.

%HANADLES situations where the chunklet is of size 1.

function [CenteredChunklets]= centerChunklets(DataSet, ChunkletSizes)

[rows cols]= size(DataSet);

numOfChunklets= length(ChunkletSizes);
i=1;

for k=1:numOfChunklets
    
    if(~ChunkletSizes(k)==0)
        centroid= mean(DataSet(:,i:i+ChunkletSizes(k)-1)');
        
        DataSet(:,i:i+ChunkletSizes(k)-1)= DataSet(:,i:i+ChunkletSizes(k)-1)- ...
            (centroid'*ones(1,ChunkletSizes(k)));
    end
    
    i= i+ChunkletSizes(k);
end

CenteredChunklets= DataSet;
