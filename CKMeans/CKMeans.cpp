#include <iostream>
#include <stdlib.h>

#include "CKMeans.h"

using namespace std;

int main(int argc, char **argv){
	switch(argc){
	case 1:
	case 2:
		cout<<"Usage: CKMeans datafile number"<<endl;
		break;
	default:
		FILE *file=fopen(argv[1],"r");
		int m,n,o;
		fscanf(file,"%d",&m);
		fscanf(file,"%d",&n);
		
		//fscanf(file,"%d",&o);
		o=0;
		
		float *data[m];
		int i,j;
		if(o==0){
			for(i=0;i<m;i++){
				data[i]=new float[n];
				for(j=0;j<n;j++){
					fscanf(file,"%f",&data[i][j]);
				}
			}
		}else{
			for(i=0;i<m;i++){
				data[i]=new float[n];
				for(j=0;j<n;j++){
					data[i][j]=0;
				}
			}
			/**
			read the sparse matrix
			*/
		}
		int k=atoi(argv[2]);
		CKMeans *ckmeans=new CKMeans(data,m,n,k);
		
		ckmeans->main();
		
		for(i=0;i<ckmeans->m;i++){
			printf("%d\n",ckmeans->p[i]);
		}
		
		break;
	}
	return 0;
}
