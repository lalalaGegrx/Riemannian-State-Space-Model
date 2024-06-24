function [Y_predict,loss]=Prediction_FC(A,H,sig,m1,m2,Y)
    T=size(Y,3);
    dim_x=size(A,1);
    dim_y=size(H,2);
    X_init=eye(dim_x);
    N=50*dim_x^2;
    X_estimate=real(RiemannianParticleFilter(A,H,sig(1),sig(2),m1,m2,N,Y,X_init));
    Y_predict=zeros(dim_y,dim_y,T);
    Y_predict(:,:,1)=Y(:,:,1);
    for i=1:T-1
        X_predict=A'*X_estimate(:,:,i)*A+m1*eye(dim_x);
        Y_predict(:,:,i+1)=H'*X_predict*H+m2*eye(dim_y);
    end
    [~,loss]=RiemannianNRMSE(Y,Y_predict);
end
