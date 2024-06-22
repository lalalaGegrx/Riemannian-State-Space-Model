clear;clc;
addpath('.\utils');
addpath('.\algorithms');

%% 
% T=200;
% dim_x=4;
% dim_y=dim_x*3;
% r_range=[0.7,0.99];
% theta_range=[-pi/4,pi/4];
% A=Generate_A(dim_x,r_range,theta_range);
% H=normrnd(0,1,[dim_x,dim_y]);
% sig1=0.4;
% sig2=0.2;
% m1=0.1;
% m2=0.1;
% X_init=eye(dim_x);
% N=10*dim_x^2;
% 
% disp(['Simulation FC dimension: ',num2str(dim_y),'*',num2str(dim_y)])
% disp('Generating simulation data...')
% [X_true,Y_true]=Generate_FC(T,A,H,sig1,sig2,m1,m2,X_init);

load('./dataset/SimulationExampleTrial-1.mat')

%%
disp('Running RiemannianExpectationMaximum...')
[A_est,H_est,sig_est,m1_est,m2_est]=RiemannianExpectationMaximum(Y_true,dim_x,N,X_init);

disp('Predicting FC...')
[Y_predict,loss]=Prediction_FC(A_est,H_est,sig_est,m1_est,m2_est,Y);

rmpath('.\utils')
rmpath('.\algorithms')
