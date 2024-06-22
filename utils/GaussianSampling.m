function r=GaussianSampling(N,n,var)

    r=normrnd(0,var,[n,N]);
    r=exp(r);
    
end
