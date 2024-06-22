function X_est_pf=RiemannianParticleFilter(A,H,sig1,sig2,m1,m2,N,Y,X_init)
    T=size(Y,3);
    dim_x=size(A,1);
    dim_y=size(H,2);
    X_est_pf=zeros(dim_x,dim_x,T);
    X_est_pf(:,:,1)=X_init;
    samples=zeros(dim_x,dim_x,N);
    for i=1:N
        samples(:,:,i)=eye(dim_x);
    end
    weights=ones(1,N)/N;

    for i=2:T
        % Predict
        eig_val=sample_by_gaussion(N,dim_x,sig1);
        for j=1:N
            S=normrnd(0,1,[dim_x,dim_x]);
            [eig_vec,~]=qr(S);
            center_l=sqrtm(A'*samples(:,:,j)*A+m1*eye(dim_x));
            samples(:,:,j)=center_l'*eig_vec'*diag(eig_val(:,j))*eig_vec*center_l;
        end
        
        % Update
        for j=1:N
            pro=exp(-distance_riemann(Y(:,:,i),H'*samples(:,:,j)*H+m2*eye(dim_y))^2/(2*sig2^2));
            weights(j)=weights(j)*pro;
        end
        w_sum=sum(weights);
        if w_sum==0
            weights=ones(1,N)/N;
        else
            weights=weights/w_sum;
            [samples,weights]=resample(N,dim_x,samples,weights);
        end
        
        % Estimate state
        try
            [X_est_pf(:,:,i),flag]=riemann_mean(samples);
        catch
            disp('Problem appear during Riemann mean');
            flag=1;
        end
        if flag==1
            try
                X_est_pf(:,:,i)=real(riemann_mean(X_est_pf(:,:,1:i-1)));
            catch
                X_est_pf(:,:,i)=real(sum(X_est_pf(:,:,1:i-1),3)/(i-1));
            end
            for j=1:N
                samples(:,:,j)=X_est_pf(:,:,i);
            end
        end
    end

    function [samples,weights]=resample(N,n,samples,weights)
        samples_new=zeros(n,n,N);
        w_cumsum=cumsum(weights);
        for l=1:N
            threshold=rand;
            for k=1:N
                if threshold<w_cumsum(k)
                    samples_new(:,:,l)=samples(:,:,k);
                    break;
                end
            end
        end
        samples=samples_new;
        weights=ones(1,N)/N;
    end
end
