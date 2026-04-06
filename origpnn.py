#----------------------------------------------------------
# origpnn.py: evaluate using original PNN
#----------------------------------------------------------
import argparse as ap,ctypes as ct,numpy as np
import os,signal,sys,time,re,glob; from pprint import pprint
#--------------------------------------------------------------
"""
DataSets={0:'abalone',1:'ionosphere',2:'isolet',3:'letter-recognition',
  4:'mnist',5:'optdigits',6:'pendigits',7:'sat',8:'segmentation',9:'mnist100',
  10:'coil-100',11:'cifar100'}
"""
DataSets={0:'abalone',1:'ionosphere',2:'isolet',3:'letter-recognition',
  4:'mnist',5:'optdigits',6:'pendigits',7:'sat',8:'segmentation'}
n=int(input(f"Input the num for DataSet to use ({DataSets}): "))
sDataSet=DataSets[n]
#-----------------------------------------------------
# No.0) read d_max.txt and extract sigma for sDataSet
#-----------------------------------------------------
_Dmax=np.loadtxt('./d_max.txt',delimiter=',',dtype=[('dataset','U100'),('value',float)])
Dmax=dict(_Dmax);
#-------------------------------------------------
# No.1) main()
#-------------------------------------------------
def main():
  global train_x,train_y,test_x,test_y,Nc,sigma
  #-------------------------------------------------
  # 1. Load Data
  #-------------------------------------------------
  train_x,train_y,test_x,test_y=import_dataset('./datasets/'+sDataSet)
  Nc=max(train_y)+1 # bit fragile ...
  sigma=Dmax[sDataSet]/Nc
  train_x,test_x=normalization(train_x,test_x,mode='-11')
  #-------------------------------------------------
  train_x_p,Mx,Nx=npMatrixToC(train_x)
  train_y_p,My,Ny=npMatrixToC(train_y)
  #------------------------------
  test_x_p,P,Q=npMatrixToC(test_x)
  #-------------------------------------------------
  # 2. Evaluate the original PNN
  #-------------------------------------------------
  signal.signal(signal.SIGINT,signal.SIG_DFL)
  print("Testing Original PNN ----"); tic()
  _y=lib.evaluatePNN(train_x_p,train_y_p,test_x_p,Mx,P,Nx,Nc,sigma)
  toc("Testing")
  #-------------------------
  y=[r for r in _y[:P]]
  confmat=np.zeros([Nc,Nc])
  for i in range(P):
    confmat[test_y[i],y[i]]=confmat[test_y[i],y[i]]+1
  c_rate=100*sum(np.diag(confmat))/len(test_y)
  print(f"Acc. {c_rate:.2f}%")
  lib.free1DInt(_y);
  signal.signal(signal.SIGINT,signal.default_int_handler)
#-------------------------------------------------
# end of main()
#-------------------------------------------------
#-------------------------------------------------
# No.2) Definitions of functions and classes
#-------------------------------------------------
c_int_p=ct.POINTER(ct.c_int); c_double_p=ct.POINTER(ct.c_double) 
_dp=np.ctypeslib.ndpointer(dtype=np.uintp,ndim=1,flags='C')
# ==============================================================
# numpy用の割り算関数(除数が0のときは0を演算結果として返す)
# ==============================================================
def user_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=(b != 0),casting='unsafe')
#-------------------------------------------------
def import_dataset(sPath):
    # CSVから読み込みとデータ形成
    trName=str(sPath)+"_tr.csv"
    tsName=str(sPath)+"_ts.csv"
    trData=np.loadtxt(trName,delimiter=',')
    tsData=np.loadtxt(tsName,delimiter=',')
    #-------------------------------------
    train_y=trData[:,0].astype(np.int32)
    test_y=tsData[:,0].astype(np.int32)
    train_x=trData[:,1:]
    test_x=tsData[:,1:]
    #-------------------------------------
    return train_x,train_y,test_x,test_y
# ==============================================================
# 学習データの値をもとに正規化を行う関数
#   mode: '01'   ->  [0, 1]の範囲に正規化
#   mode: '-11'  ->  [-1, 1]の範囲に正規化
# ==============================================================
def normalization(train_x,test_x,mode='01'):
    tr_max=np.max(train_x,axis=0)
    tr_min=np.min(train_x,axis=0)
    min_max_range=tr_max-tr_min
    if mode=='01':
        train_x=user_divide(train_x-tr_min,min_max_range)
        test_x=user_divide(test_x-tr_min,min_max_range)
    elif mode=='-11':
        train_x=(user_divide(train_x-tr_min, min_max_range)-0.5)*2
        test_x=(user_divide(test_x-tr_min, min_max_range)-0.5)*2
    return train_x,test_x
#-------------------------------------------
def tic():
    #require to import time
    global start_time_tictoc
    start_time_tictoc=time.time()
#-------------------------------------------
def toc(tag="elapsed time"):
    if "start_time_tictoc" in globals():
        print("{}: {:.4f} [sec]".format(tag, time.time() \
					- start_time_tictoc))
    else:
        print("tic has not been called")
#-------------------------------------------
def npMatrixToC(arr):
  if len(arr.shape)==1:
    _arr=arr.astype('double')
    M=arr.shape[0]; N=1
    retp=_arr.ctypes.data_as(c_double_p)
  else:
    if arr.dtype=='float64':
      _arr=arr
    else:
      _arr=arr.astype('double')
    M=arr.shape[0]; N=arr.shape[1]
    retp=(_arr.__array_interface__['data'][0]
      +np.arange(M)*_arr.strides[0]).astype(np.uintp)
  return retp,M,N
#-------------------------------------------------
# No.3) Init. before main()
#-------------------------------------------------
lib=ct.CDLL('./origpnn.so')
#--------------------------------------
lib.free1DInt.argtypes=(c_int_p,)
lib.free1DInt.restype=None
#--------------------------------------
lib.evaluatePNN.argtypes=(_dp,c_double_p,_dp,ct.c_int,
	ct.c_int,ct.c_int,ct.c_int,ct.c_double)
lib.evaluatePNN.restype=c_int_p
#-------------------------------------------------
# No.4) Finally, call main() appeared on the top
#-------------------------------------------------
if __name__=='__main__':
  main()
