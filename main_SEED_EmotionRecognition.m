clear;clc;
addpath('.\utils');
addpath('.\algorithms');

%%
load('./dataset/SEEDExampleSession-1.mat')
dim_x=6;
channels=12;
N=10*dim_x^2;
X_init=eye(dim_x);
T_trial=175;

%%
disp('Running RiemannianExpectationMaximum Algorithm ...')
[A_est,H_est,sig_est,m1_est,m2_est]=RiemannianExpectationMaximum(Y,dim_x,N,X_init);
X_estimation=RiemannianParticleFilter(A_est,H_est,sig_est(1),sig_est(2),m1_est,m2_est,N,Y,X_init);

X_vector=zeros(T-1,(channels*(channels+1))/2);
for i=2:T
    L=tril(X_estimation(:,:,i));
    X_vector(i-1,:)=L(L~=0);
end
y=nominal(labels);

SVMModel=fitcsvm(X_vector,y,'KernelFunction','linear');
CVSVMModel=crossval(SVMModel);
loss_cv=kfoldLoss(CVSVMModel);

rmpath('.\utils')
rmpath('.\algorithms')
