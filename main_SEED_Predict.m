clear;clc;
addpath('.\utils');
addpath('.\algorithms');

%%
load('./dataset/SEEDExampleTrial-1.mat')
dim_x=6;
N=10*dim_x^2;
X_init=eye(dim_x);

%%
disp('Running RiemannianExpectationMaximum ...')
[A_est,H_est,sig_est,m1_est,m2_est]=RiemannianExpectationMaximum(Y,dim_x,N,X_init);

disp('Predicting FC...')
[Y_predict,loss]=Prediction_FC(A_est,H_est,sig_est,m1_est,m2_est,Y);

rmpath('.\utils')
rmpath('.\algorithms')
