%% ------------------------ Author: Maryam Bijanzadeh  ----------------


fileID = fopen('/tmp/RF_ThreeClass_DecoderRandom.txt','r'); 
params = textscan(fileID,'%s'); 

% Functions and subfunction should be added to the main path 

% DIR = Navigate to a directory where you have the fetaures : 
% and load the matfile containing features e.g.:
% Pos_Feature : frequency  x positve samples x channel number
% Pos_Feature_Reg : frequency  x positve samples x region number

msg = sprintf('About to start running with EC:  %s', cell2char(params{1}));
msg



Itr_Num = size(Feature_Reg,1) ;    Fold_Num = size(RF_Reg{1},2) ;          
RF_Random_Full = cell(Itr_Num,1);  RF_Random_Reg = cell(Itr_Num,1);  
Num = size(Feature{1},1)/3; % we have 3 classes 
Label = [zeros(Num,1); ones(Num,1);ones(Num,1)*2 ]; %  neutral = 0 ,Pos labels = 1 , Neg Labels =2 

poolobj = gcp('nocreate');
delete(poolobj);
G= parpool(5); 


    parfor Itr= 1:Itr_Num 
        fprintf('Dataset # = %d \n',Itr)

        [RF_Random, RF_Region_Random]= ShuffledModel_Kfold_RF_ThreeClass(Label,Feature{Itr},Fold_Num,Feature_Reg{Itr});   

        RF_Random_Full{Itr} = RF_Random;     RF_Random_Reg{Itr} = RF_Region_Random;    

    end 

save([DIR,'Random_RF_ThreeClass'],'RF_Random_Full','RF_Random_Reg','Fold_Num','Feature','Feature_Reg','-v7.3')
