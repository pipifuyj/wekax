#include <cmath>

class CKMeans{
public:
	float **data;
	int k;
	
	int m;
	int n;
	
	int *means;
	
	int *p;
	
	float *sims;
	
	CKMeans(float **data, int m, int n, int k);
	float sim(int i,int j);
	float crfun(int cluster, int mean);
	void initMeans();
	int getMean(int cluster);
	bool setMeans();
	int main();
};

CKMeans::CKMeans(float **_data, int _m, int _n, int _k){
	data=_data;
	k=_k;
	
	m=_m;
	n=_n;
	
	means=new int[k];
	
	p=new int[m];
	
	sims=new float[m*m];
	
	int i,j;
	for(i=0;i<m;i++){
		for(j=0;j<m;j++){
			sims[i*m+j]=-1;
		}
	}
}

float CKMeans::sim(int i,int j){
	int ij=i*m+j, l;
	if(sims[ij]>=0)return sims[ij];
	float II=0, JJ=0, IJ=0;
	for(l=0; l<n; l++){
		II+=data[i][l]*data[i][l];
		JJ+=data[j][l]*data[j][l];
		IJ+=data[i][l]*data[j][l];
	}
	sims[ij]=IJ/sqrt(II*JJ);
	//printf("II, JJ, IJ=%f, %f, %f\n", II, JJ, IJ);
	//printf("sims[%d]=%f\n",ij,sims[ij]);
	return sims[ij];
}

float CKMeans::crfun(int cluster, int mean){
	int i;
	float cr=0;
	for(i=0;i<m;i++){
		if(p[i]==cluster){
			cr+=sim(i,mean);
		}
	}
	cr=sqrt(cr);
	return cr;
}

void CKMeans::initMeans(){
	int i;
	for(i=0;i<k;i++){
		means[i]=i;
	}
}

int CKMeans::getMean(int cluster){
	int mean,i;
	float cr,t;
	for(i=0;i<m;i++){
		if(p[i]==cluster)break;
	}
	mean=i;
	cr=crfun(cluster,mean);
	for(i;i<m;i++){
		if(p[i]==cluster){
			t=crfun(cluster,i);
			if(t>cr){
				mean=i;
				cr=t;
			}
		}
	}
	return mean;
}

bool CKMeans::setMeans(){
	int i,j,intt;
	float s,floatt;
	for(i=0;i<m;i++){
		intt=0;
		s=sim(i,means[intt]);
		for(j=0;j<k;j++){
			floatt=sim(i,means[j]);
			if(floatt>s){
				intt=j;
				s=floatt;
			}
		}
		if(p[i]!=intt){
			p[i]=intt;
			//printf("+->p[%d]=%d\n",i,intt);
		}else{
			//printf("-->p[%d]=%d\n",i,intt);
		}
	}
	bool flag=false;
	for(i=0;i<k;i++){
		intt=getMean(i);
		if(means[i]!=intt){
			means[i]=intt;
			flag=true;
			//printf("means[%d]=%d\n",i,intt);
		}
	}
	return flag;
}

int CKMeans::main(){
	initMeans();
	while(setMeans()){
		//printf("-+-+-+-+-+-+-+-+-+-+\n");
	}
	return 0;
}