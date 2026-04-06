#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#   define API __declspec(dllexport)
#else
#   define API
#endif

extern "C"{

API int *evaluatePNN(double **x_tr,double *y_tr,double **x_ts,
	int nTr,int nTs,int nVecLen,int nClasses,double sigma){
	double *vOUnits=(double *)calloc(nClasses,sizeof(double));
	int *nOUnits=(int *)calloc(nClasses,sizeof(int));
  int *y=(int *)calloc(nTs,sizeof(int)); // the output class IDs obtained from the CS-PNN
  for(int i=0;i<nClasses;i++)nOUnits[i]=0;
  for(int i=0;i<nTs;i++){
	  for(int j=0;j<nClasses;j++)vOUnits[j]=0.0;
    for(int j=0;j<nTr;j++){
			double t=0.0;
      for(int k=0;k<nVecLen;k++)t+=(x_ts[i][k]-x_tr[j][k])*(x_ts[i][k]-x_tr[j][k]);
      int n=(int)y_tr[j];
      vOUnits[n]+=exp(-t/(sigma*sigma));
      if(i==0)nOUnits[n]++;
    }
    for(int j=0;j<nClasses;j++)vOUnits[j]/=(double)nOUnits[j];
    double max_v=vOUnits[0],max_j=0;
    for(int j=1;j<nClasses;j++)
      if(max_v<vOUnits[j])max_v=vOUnits[j],max_j=j;
    y[i]=max_j;
  }
  free(vOUnits);free(nOUnits);
	return y;
}
//-----------------------------------------------------
void free1DInt(int *data){
  free(data);
}

}
