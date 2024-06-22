function mat=Generate_A(dim,r_range,theta_range)
    cMat=zeros(dim,1);
    for i=1:2:dim-mod(dim,2)
        r=r_range(1)+rand(1)*diff(r_range);
        theta=theta_range(1)+rand(1)*diff(theta_range);
        cMat(i)=r*exp(1i*theta);
        cMat(i+1)=r*exp(-1i*theta);
    end
    if mod(dim,2)
        cMat(dim)=r_range(1)+rand(1)*diff(r_range);
    end
    D=cMat;
    realDform=comp2rdf(diag(D));
    mat=realDform;
    P=normrnd(0,0.1,[dim,dim]);
    mat=P*realDform/P;
end

function dd=comp2rdf(d)
    i=find(imag(diag(d))');
    index=i(1:2:length(i));
    if isempty(index)
        dd=d;
    elseif (max(index)==size(d,1)) | any(conj(d(index,index))~=d(index+1,index+1))
            error(message('Complex conjugacy not satisfied'));
    end
    j = sqrt(-1);
    t = eye(length(d));
    twobytwo = [1 1;j -j];
    for i=index
        t(i:i+1,i:i+1) = twobytwo;
    end
    dd=t*d/t;
end