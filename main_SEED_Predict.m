clear;clc;
addpath('.\utils');
addpath('.\algorithms');

%%
load('./dataset/SEEDExampleTrial-2.mat')
[~,dim_y,T]=size(Y);
dim_x=8;
N=10*dim_x^2;
X_init=eye(dim_x);
y_ref=5*eye(dim_y);

%%
disp('Running RiemannianExpectationMaximum ...')
[A_est,H_est,sig_est,m1_est,m2_est]=RiemannianExpectationMaximum(Y,dim_x,N,X_init);

disp('Predicting FC...')
[Y_predict,loss]=Prediction_FC(A_est,H_est,sig_est,m1_est,m2_est,Y);

%%
figure(1)
d_true=zeros(1,T);
d_predict=zeros(1,T);
for i=1:T
    d_true(i)=distance_riemann(Y(:,:,i),y_ref);
    d_predict(i)=distance_riemann(Y_predict(:,:,i),y_ref);
end
hold on;
plot(1:T,d_true,'linewidth',2);
plot(1:T,d_predict,'linewidth',2);
xlabel('Time steps')
ylabel('Riemannian distance')

rmpath('.\utils')
rmpath('.\algorithms')
