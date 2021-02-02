%% ------------------------ Author: Maryam Bijanzadeh  ----------------


fileID = fopen('/tmp/RF_EN_DecoderRandom.txt','r'); 
params = textscan(fileID,'%s \n %s \n %s'); 


% Functions and subfunction should be added to the main path 

% DIR = Navigate to a directory where you have the fetaures : 
% and load the matfile containing features e.g.:
% Pos_Feature : frequency  x positve samples x channel number
% Pos_Feature_Reg : frequency  x positve samples x region number
msg = sprintf('About to start running with EC:  %s and Affect_Type: %s and %s ', cell2char(params{1}),cell2char(params{2}),cell2char(params{3}));
msg


rng(19840901);

Affect_Type = {cell2char(params{2}),cell2char(params{3})}  


Itr_Num = size(Feature_Reg,1) ;    Fold_Num = size(RF_Reg{1},2) ;          

RF_Random_Full = cell(Itr_Num,1);  RF_Random_Reg = cell(Itr_Num,1);  

     poolobj = gcp('nocreate');
     delete(poolobj);
     G= parpool(5); 


       
        Label = [ones(size(Feature{1},1)/2,1); zeros(size(Feature{1},1)/2,1)];
        Num = size(Feature{1},1)/2; 
         
        parfor Itr= 1:Itr_Num
            
            fprintf('Dataset # = %d \n',Itr)
         
            [RF_Random,  RF_Region_Random]= ShuffledModel_Kfold_RF_PCA(Label,Feature{Itr},Fold_Num+1,Feature_Reg{Itr});   
            
            RF_Random_Full{Itr} = RF_Random;     RF_Random_Reg{Itr} = RF_Region_Random;    
            
        end 
          


save([DIR,'Random_RF_100D' Affect_Type{1} '_' Affect_Type{2}],'RF_Random_Full','RF_Random_Reg','Affect_Type','Fold_Num',...
    'Feature','Feature_Reg','Feat_Label','Feat_Label_Reg','-v7.3')
% 
