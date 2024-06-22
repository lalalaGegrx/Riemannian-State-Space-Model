function

subjects=1;
for s=5
    load(['D:\matlab\bin\mainfold\EM\EM_HCI\Result_HCI_valence_RSSM\Subject-',num2str(s),'.mat']);
    
    [channels,~,T]=size(X_est_em);
    trials=length(Label_valence);
    
    % label
    samples_per_trial=ones(1,trials)*45;
    for idx=idx_ban_raw
        place=ceil(idx/45);
        samples_per_trial(place)=samples_per_trial(place)-1;
    end
    samples_label=[];
    for i=1:trials
        if Label_valence(i)>5
            samples_label=[samples_label,ones(1,samples_per_trial(i))*1];
        else
            samples_label=[samples_label,ones(1,samples_per_trial(i))*-1];
        end
    end
    samples_label=samples_label(2:T);
    y=nominal(samples_label);
    
    % data
    Y_vector=zeros(T-1,(channels*(channels+1))/2);
    for i=2:T
        L=tril(X_est_em(:,:,i));
        Y_vector(i-1,:)=L(L~=0);
    end
    
    SVMModel=fitcsvm(Y_vector,y,'KernelFunction','linear');
    CVSVMModel=crossval(SVMModel);
    loss_cv=kfoldLoss(CVSVMModel);
end
