function [A_est,H_est,sig_est,m1_est,m2_est]=RiemannianExpectationMaximum(Y,dim_x,N,X_init)
    load('.\utils\RiemannianGaussianApproximation.mat');

    [dim_y,~,T]=size(Y);
    max_iter=18;
    max_iter_grad=1e3;
    mini_state=1e-2;
    mini_measure=1e-1;
    lr_state=1e-2;
    lr_measure=1e-2;
    m1_secure=[0.01,2];
    m2_secure=[0.01,2];
    sig1_secure=[0.05,0.9];
    sig2_secure=[0.05,0.9];
    eps=1e-5;
    T_start=50;

    A_est=eye(dim_x)*0.99;
    H_est=normrnd(0,1,[dim_x,dim_y]);
    m1_est=0.1;
    m2_est=0.1;
    sig_est=[0.5,0.5];
    for iter_em=1:max_iter
        % E step
        X_est_pf=RiemannianParticleFilter(A_est,H_est,sig_est(1),sig_est(2),m1_est,m2_est,N,Y,X_init);

        % M step
        cost_state_last=1e100;
        cost_measure_last=1e100;
        cost_state=1e10;
        cost_measure=1e10;
        mome_m1=0;
        mome_m2=0;
        mome_A=zeros(dim_x);
        mome_H=zeros(dim_x,dim_y);
        v_m1=0;
        v_m2=0;
        v_A=zeros(dim_x);
        v_H=zeros(dim_x,dim_y);
        beta1=0.9;
        beta2=0.99;
        if mod(iter_em,2)==0
            for iter_grad=1:max_iter_grad
                if abs(cost_state-cost_state_last)>mini_state
                    cost_state_last=cost_state;
                    cost_state=0;
                    for k=T_start:T-1
                        cost_state=cost_state+distance_riemann(X_est_pf(:,:,k+1),A_est'*X_est_pf(:,:,k)*A_est+m1_est*eye(dim_x))^2;
                    end
                    if imag(cost_state)~=0
                        break;
                    end
                    if cost_state>cost_state_last
                        lr_state=lr_state*0.9;
                    end
                    [grad_m1,grad_A]=caculate_numgrad_state(T_start,T,X_est_pf,A_est,m1_est,cost_state);
                    mome_m1=beta1*mome_m1+(1-beta1)*grad_m1;
                    mome_A=beta1*mome_A+(1-beta1)*grad_A;
                    v_m1=beta2*v_m1+(1-beta2)*grad_m1*grad_m1;
                    v_A=beta2*v_A+(1-beta2)*(grad_A.*grad_A);
                    mome_m1_hat=mome_m1/(1-beta1^iter_grad);
                    mome_A_hat=mome_A/(1-beta1^iter_grad);
                    v_m1_hat=v_m1/(1-beta2^iter_grad);
                    v_A_hat=v_A/(1-beta2^iter_grad);

                    A_est=A_est-lr_state*mome_A_hat./(sqrt(v_A_hat)+eps);
                    m1_tem=m1_est-lr_state*mome_m1_hat/(sqrt(v_m1_hat)+eps);
                    if m1_tem>m1_secure(1)&&m1_tem<m1_secure(2)
                        m1_est=m1_tem;
                    end
                end
            end
            
            e1=zeros(1,T-1);
            for i=T_start:T-1
                y=A_est'*X_est_pf(:,:,i)*A_est+m1_est*eye(dim_x);
                e1(i)=distance_riemann(X_est_pf(:,:,i+1),y)^2;
            end
            sig1_est=sqrt(sum(e1)/(dim_x*(T-T_start)));
            
            e2=zeros(1,T);
            for i=T_start:T
                y=H_est'*X_est_pf(:,:,i)*H_est+m2_est*eye(dim_y);
                e2(i)=distance_riemann(Y(:,:,i),y)^2;
            end
            sig2_est=sqrt(sum(e2)/(dim_y*(T-T_start+1)));
            
            % sig_tem=[sig1_est,sig2_est]*matrix_approximation_simulation;
            sig_tem=[sig1_est,sig2_est]*matrix_approximation_seed;
            if sig_tem(1)>sig1_secure(1)&&sig_tem(1)<sig1_secure(2)
                sig_est(1)=sig_tem(1);
            end
            if sig_tem(2)>sig2_secure(1)&&sig_tem(2)<sig2_secure(2)
                sig_est(2)=sig_tem(2);
            end
        else
            for iter_grad=1:max_iter_grad
                if abs(cost_measure-cost_measure_last)>mini_measure
                    cost_measure_last=cost_measure;
                    cost_measure=0;
                    for k=T_start:T
                        cost_measure=cost_measure+distance_riemann(Y(:,:,k),H_est'*X_est_pf(:,:,k)*H_est+m2_est*eye(dim_y))^2;
                    end
                    cost_measure
                    if imag(cost_measure)~=0
                        break;
                    end
                    if cost_measure>cost_measure_last
                        lr_measure=lr_measure*0.9;
                    end
                    [grad_m2,grad_H]=caculate_numgrad_observation(T_start,T,X_est_pf,Y,A_est,H_est,m2_est,cost_measure);
                    mome_m2=beta1*mome_m2+(1-beta1)*grad_m2;
                    mome_H=beta1*mome_H+(1-beta1)*grad_H;
                    v_m2=beta2*v_m2+(1-beta2)*grad_m2*grad_m2;
                    v_H=beta2*v_H+(1-beta2)*(grad_H.*grad_H);
                    mome_m2_hat=mome_m2/(1-beta1^iter_grad);
                    mome_H_hat=mome_H/(1-beta1^iter_grad);
                    v_m2_hat=v_m2/(1-beta2^iter_grad);
                    v_H_hat=v_H/(1-beta2^iter_grad);

                    H_est=H_est-lr_measure*mome_H_hat./(sqrt(v_H_hat)+eps);
                    m2_tem=m2_est-lr_measure*mome_m2_hat/(sqrt(v_m2_hat)+eps);
                    if m2_tem>m2_secure(1)&&m2_tem<m2_secure(2)
                        m2_est=m2_tem;
                    end
                end
            end
        end
    end

end

