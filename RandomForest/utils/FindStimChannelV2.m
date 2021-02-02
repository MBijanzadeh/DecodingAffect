
function StimChan=FindStimChannelV2(ChanName,anatomy,CellIdx,flag)
% ChanName is in istring
% You need to load clinical_elecs_all before runing this


if flag == 0
    StimChan=[];
    for i=1:size(ChanName,1)
        
        
        A=[ strfind(anatomy,ChanName{i,CellIdx})];
        if ~isempty(A)
            StimChan=[StimChan i] ;
            
        end
        
    end
    
else
    

StimChan=[];
for i=1:size(anatomy,1)
    
    
    A=[ strfind(anatomy{i,CellIdx},ChanName)];
    if ~isempty(A)
        StimChan=[StimChan i] ;
        
    end
    
end

end