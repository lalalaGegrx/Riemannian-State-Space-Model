clear;clc;

subjects=35;
loss_cv=zeros(1,subjects);
for s=1:subjects
    load(['.\Result\Data_S',num2str(s),'.mat']);
    Data_X=Data_X(:,:,10:end,:,:);
    [channels,~,T,bp,trials]=size(Data_X);
    data=zeros(3*channels,3*channels,T*trials);
    Data_X=permute(Data_X,[1,2,4,3,5]);
    Data_X=reshape(Data_X,[channels,channels,bp,T*trials]);
    data(1:channels,1:channels,:)=Data_X(:,:,1,:);
    data(channels+1:2*channels,channels+1:2*channels,:)=Data_X(:,:,2,:);
    data(2*channels+1:3*channels,2*channels+1:3*channels,:)=Data_X(:,:,3,:);
    
    samples_label=[];
    for i=1:trials
        samples_label=[samples_label;ones(T,1)*label(i)];
    end

    X_vector=zeros(T*trials,3*channels*(channels+1)/2);
    for i=1:T*trials
        L=tril(data(:,:,i));
        X_vector(i,:)=L(L~=0);
    end
    
    SVMModel=fitcecoc(X_vector,samples_label);
    CVSVMModel=crossval(SVMModel);
    loss_cv(s)=kfoldLoss(CVSVMModel);
end

acc_cv=(1-loss_cv)*100;
acc_mean=mean(acc_cv);
sem=sqrt(sum((acc_cv-acc_mean).^2)/(subjects));
