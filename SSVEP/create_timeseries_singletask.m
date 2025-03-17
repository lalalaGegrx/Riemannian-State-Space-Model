function [Y,label]=create_timeseries_singletask(Data,event_pos,event_type)

    trials=31;
    channels=8;
    sr=256;
    trial_time=3;
    window=2;
    N_overlap=16;
    Y=zeros(channels,channels,(trial_time-window)*N_overlap,24);
    label=[];
    l=1;
    for tr=1:trials
        if event_type(3*tr+2)~=33024
            t=event_pos(3*tr)+2*sr;
            label(l)=event_type(3*tr+2)-33025;
            i=1;
            while t<event_pos(3*tr)+3*sr
                data_seg=Data(:,t:t+window*sr);
                t=t+sr/16;
                Y(:,:,i,l)=covariances(data_seg);
                %Y(:,:,end+1)=corr(data_seg');
                i=i+1;
            end
            l=l+1;
        end
    end
    
end