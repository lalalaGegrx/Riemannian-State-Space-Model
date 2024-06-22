function [loss,loss_final]=RiemannianNRMSE(X,X_estimate)
    T=size(X,3);
    x_center=riemann_mean(X);
    
    d_true=zeros(1,T);
    d_estimate=zeros(1,T);
    loss=zeros(1,T);
    for i=1:T
        d_true(i)=distance_riemann(X(:,:,i),x_center);
        d_estimate(i)=distance_riemann(X(:,:,i),X_estimate(:,:,i));
        loss(i)=sum(d_estimate(1:i).^2)/sum(d_true(1:i).^2);
    end
    loss_final=sum(d_estimate(100:T).^2)/sum(d_true(100:T).^2);

end
