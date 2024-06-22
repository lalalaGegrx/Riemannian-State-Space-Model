function [grad_m,grad_H]=caculate_numgrad_observation(T_start,T,X,Y,A,H,m2,cost)

    eps=1e-5;
    dim_x=length(A);
    dim_y=length(H);
    cost_m=0;
    for k=T_start:T
        cost_m=cost_m+distance_riemann(Y(:,:,k),H'*X(:,:,k)*H+(m2+eps)*eye(dim_y))^2;
    end
    grad_m=(cost_m-cost)/eps;

    grad_H=zeros(dim_x,dim_y);
    for i=1:dim_x
        for j=1:dim_y
            H_t=H;
            H_t(i,j)=H_t(i,j)+eps;
            cost_H=0;
            for k=T_start:T
                cost_H=cost_H+distance_riemann(Y(:,:,k),H_t'*X(:,:,k)*H_t+m2*eye(dim_y))^2;
            end
            grad_H(i,j)=(cost_H-cost)/eps;
        end
    end

end
