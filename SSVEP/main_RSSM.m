clear;clc;

F_1=designfilt('bandpassiir','FilterOrder',20, ...
         'HalfPowerFrequency1',12.9,'HalfPowerFrequency2',13.1, ...
         'SampleRate',256);
F_2=designfilt('bandpassiir','FilterOrder',20, ...
         'HalfPowerFrequency1',16.9,'HalfPowerFrequency2',17.1, ...
         'SampleRate',256);
F_3=designfilt('bandpassiir','FilterOrder',20, ...
         'HalfPowerFrequency1',20.9,'HalfPowerFrequency2',21.1, ...
         'SampleRate',256);

subjects=12;
trials=24;
num_channels=8;
dim_x=5;
num_file=1;
for s=1:subjects
    disp(['Subejct-',num2str(s)])
    folder=['.\Dataset_SSVEP\S',num2str(s),'\'];
    files=dir(fullfile(folder,'*.mat'));
    for f=1:numel(files)
        disp(['File-',num2str(f)])
        filename=files(f).name;
        load([folder,filename]);
        event_pos=double(int32((event_pos)));
        event_type=double(int32((event_type)));
        
        Data_X=zeros(dim_x,dim_x,16,3,trials);
        Data_Y=zeros(num_channels,num_channels,16,3,trials);
        for bp=1:3
            eval(['F_bp=F_',num2str(bp),';']);
            Data_bp=zeros(size(Data));
            for c=1:num_channels
                Data_bp(c,:)=filtfilt(F_bp,Data(c,:));
            end
            [Y_alltrial,label]=create_timeseries_singletask(Data_bp,event_pos,event_type);
            T=size(Y_alltrial,3);

            for t=1:trials
                Y=Y_alltrial(:,:,:,t);
                %for i=1:num_channels
                %    Y(i,i,:)=Y(i,i,:)+1e-4;
                %end
                y_center=riemann_mean(Y);
                for i=1:T
                    Y(:,:,i)=sqrtm(y_center)\Y(:,:,i)/sqrtm(y_center)';
                end
                y_center=riemann_mean(Y);

                % EM
                dim_y=num_channels;
                N=10*dim_x^2;
                m1_secure=[0.001,0.8];

                max_iter=12;
                max_iter_grad=5e6;
                mini_state=1e-4;
                mini_measure=1e-1;
                lr_state=3e-2;
                lr_measure=1e-2;
                eps=1e-5;

                r_range=[0.7,0.99];
                theta_range=[-pi/4,pi/4];
                % X_init=eye(dim_x);
                X_init=Y(1:dim_x,1:dim_x,1);
                A_est=eye(dim_x)*0.99;
                H_est=normrnd(0,1,[dim_x,dim_y]);
                m1_est=0.1;
                m2_est=0.1;
                sig_est=[0.8,0.1];
                T_start=1;
                break_flag=0;

                for iter_em=1:max_iter
                    if(break_flag==1)
                        break;
                    end

                    if(iter_em>1)
                        mini_measure=1e-3;
                        lr_state=3e-3;
                        lr_measure=3e-2;
                    end

                    X_est_pf=real(PF(T,A_est,H_est,sig_est(1),sig_est(2),m1_est,m2_est,N,Y));

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

                    if(mod(iter_em,3)==1)
                        for iter_grad=1:max_iter_grad
                            if(abs(cost_measure-cost_measure_last)>mini_measure)
                                cost_measure_last=cost_measure;

                                cost_measure=0;
                                for k=T_start:T
                                    cost_measure=cost_measure+distance_riemann(Y(:,:,k),H_est'*X_est_pf(:,:,k)*H_est+m2_est*eye(dim_y))^2;
                                end
                                if(imag(cost_measure)~=0)
                                    break_flag=1;
                                    break;
                                end
                                if(cost_measure>cost_measure_last)
                                    lr_measure=lr_measure*0.9;
                                end
                                [grad_m2,grad_H]=caculate_numgrad_measure(T_start,T,X_est_pf,Y,A_est,H_est,m2_est,cost_measure);

                                mome_m2=beta1*mome_m2+(1-beta1)*grad_m2;
                                mome_H=beta1*mome_H+(1-beta1)*grad_H;
                                v_m2=beta2*v_m2+(1-beta2)*grad_m2*grad_m2;
                                v_H=beta2*v_H+(1-beta2)*(grad_H.*grad_H);
                                mome_m2_hat=mome_m2/(1-beta1^iter_grad);
                                mome_H_hat=mome_H/(1-beta1^iter_grad);
                                v_m2_hat=v_m2/(1-beta2^iter_grad);
                                v_H_hat=v_H/(1-beta2^iter_grad);

                                H_est=H_est-lr_measure*mome_H_hat./(sqrt(v_H_hat)+eps);
                                m2_est=m2_est-lr_measure*mome_m2_hat/(sqrt(v_m2_hat)+eps);
                            end
                        end
                    elseif(mod(iter_em,3)==2)
                        for iter_grad=1:max_iter_grad
                            if(abs(cost_state-cost_state_last)>mini_state)
                                cost_state_last=cost_state;

                                cost_state=0;
                                for k=T_start:T-1
                                    cost_state=cost_state+distance_riemann(X_est_pf(:,:,k+1),A_est'*X_est_pf(:,:,k)*A_est+m1_est*eye(dim_x))^2;
                                end
                                if(imag(cost_state)~=0)
                                    break_flag=1;
                                    break;
                                end
                                if(cost_state>cost_state_last)
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

                                A_est_new=A_est-lr_state*mome_A_hat./(sqrt(v_A_hat)+eps);
                                eig_val=eig(A_est_new);
                                val_flag=0;
                                for k=1:length(eig_val)
                                    r=real(eig_val(k))^2+imag(eig_val(k))^2;
                                    if(r>1)
                                        val_flag=1;
                                        break;
                                    end
                                end
                                if(val_flag==1)
                                    A_est=A_est_new/vrho(A_est_new);
                                else
                                    A_est=A_est_new;
                                end

                                m1_est_next=m1_est-lr_state*mome_m1_hat/(sqrt(v_m1_hat)+eps);
                                if(m1_est_next>m1_secure(1))
                                    m1_est=m1_est_next;
                                elseif(m1_est_next>m1_secure(2))
                                    m1_est=0.4;
                                else
                                    break_flag=1;
                                    break;
                                end
                            end
                        end
                    end
                end
                Data_X(:,:,:,bp,t)=X_est_pf;
                Data_Y(:,:,:,bp,t)=Y;
            end
        end
        save(['.\Result\Data_S',num2str(num_file),'.mat'],'Data_X','Data_Y','label');
        num_file=num_file+1;
    end
end

