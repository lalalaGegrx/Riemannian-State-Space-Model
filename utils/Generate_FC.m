function [X,Y]=Generate_FC(T,A,H,sig1,sig2,m1,m2,X_init)
    dim_x=size(A,1);
    dim_y=size(H,2);
    X=zeros(dim_x,dim_x,T);
    Y=zeros(dim_y,dim_y,T);
    X(:,:,1)=X_init;
    
    eig_X=GaussianSampling(T,dim_x,sig1);
    eig_X(:,1)=eig(X_init);
    for i=2:T
        S=normrnd(0,1,[dim_x,dim_x]);
        [eig_vec,~]=qr(S);
        center_l=sqrtm(A'*X(:,:,i-1)*A+m1*eye(dim_x));
        X(:,:,i)=center_l'*eig_vec'*diag(eig_X(:,i))*eig_vec*center_l;
    end

    eig_Y=GaussianSampling(T,dim_y,sig2);
    for i=1:T
        S=normrnd(0,1,[dim_y,dim_y]);
        [eig_vec,~]=qr(S);
        center_l=sqrtm(H'*X(:,:,i)*H+m2*eye(dim_y));
        Y(:,:,i)=center_l'*eig_vec'*diag(eig_Y(:,i))*eig_vec*center_l;
    end
end
