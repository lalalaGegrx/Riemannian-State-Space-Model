function [Y_predict,loss]=Prediction_FC(A_est,H_est,sig_est,m1_est,m2_est,Y)
    T=size(Y,3);
    dim_x=size(A,1);
    dim_y=size(H,2);
    N=50*dim_x^2;
    X_estimate=real(RiemannianParticleFilter(T,A_est,H_est,sig_est(1),sig_est(2),m1_est,m2_est,N,Y));
    Y_predict=zeros(dim_y,dim_y,T);
    Y_predict(:,:,1)=Y(:,:,1);
    for i=1:T-1
        X_predict=A_est'*X_estimate(:,:,i)*A_est+m1_est*eye(dim_x);
        Y_predict(:,:,i+1)=H_est'*X_predict*H_est+m2_est*eye(dim_y);
    end
    [~,loss]=RiemannianNRMSE(Y,Y_predict);
end
