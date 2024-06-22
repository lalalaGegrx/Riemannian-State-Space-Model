function [grad_m,grad_A]=caculate_numgrad_state(T_start,T,X,A,m1,cost)

    eps=1e-5;
    dim_x=length(A);
    cost_m=0;
    for k=T_start:T-1
        cost_m=cost_m+distance_riemann(X(:,:,k+1),A'*X(:,:,k)*A+(m1+eps)*eye(dim_x))^2;
    end
    grad_m=(cost_m-cost)/eps;

    grad_A=zeros(dim_x,dim_x);
    for i=1:dim_x
        for j=1:dim_x
            A_t=A;
            A_t(i,j)=A_t(i,j)+eps;
            cost_A=0;
            for k=T_start:T-1
                cost_A=cost_A+distance_riemann(X(:,:,k+1),A_t'*X(:,:,k)*A_t+m1*eye(dim_x))^2;
            end
            grad_A(i,j)=(cost_A-cost)/eps;
        end
    end

end