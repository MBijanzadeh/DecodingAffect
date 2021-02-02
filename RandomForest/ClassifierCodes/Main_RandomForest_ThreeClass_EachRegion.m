


%% ------------------------ Author: Maryam Bijanzadeh  ----------------

% Trains 3 class Random forest for Neg-pos and netural with 100
% datasets + Run it for each regions similar as EachRegion

fileID = fopen('/tmp/DecoderParams_RF_ThreeClass.txt','r'); 
params = textscan(fileID,'%s'); 


% Functions and subfunction should be added to the main path 

% DIR = Navigate to a directory where you have the fetaures : 
% and load the matfile containing features e.g.:
% Pos_Feature : frequency  x positve samples x channel number
% Pos_Feature_Reg : frequency  x positve samples x region number

msg = sprintf('About to start running with EC:  %s ', cell2char(params{1}))
% msg
% 


Num = size(Feature{1},1)/3;  Reg_counter = numel(Reg_Label); 
Label = [zeros(Num,1); ones(Num,1);ones(Num,1)*2 ]; %  neutral = 0 ,Pos labels = 1 , Neg Labels =2 


poolobj = gcp('nocreate');
delete(poolobj);
G= parpool(5); 

Itr_Num = size(Feature_Reg,1) ;    Fold_Num = size(RF_Reg{1},2) ;          

RF_Full = cell(Itr_Num,Reg_counter);  RF_Reg = cell(Itr_Num,Reg_counter);  
Med_Feature = cell(Itr_Num,Reg_counter);  Med_Feature_Reg = cell(Itr_Num,Reg_counter); 


 tic         
   for Region = 1: Reg_counter   
       
        Index_Features_Reg= FindStimChannelV2(Reg_Label{Region}, Feat_Label_Reg, 1,1); % Which indices to include Feature_Reg

        Index_Features= FindStimChannelV2(Reg_Label{Region}, Feat_Label, 1,1); % Which indices to include from the Feature_Reg

       parfor Itr= 1:Itr_Num 
                                                      
            Med_Feature{Itr,Region} = Feature{Itr}(:,Index_Features); 
            Med_Feature_Reg{Itr,Region} = Feature_Reg{Itr}(:,Index_Features_Reg);
            
            fprintf('Dataset # = %d Region = %d \n',Itr,Region )
               
            [RF_Full_Model, RF_Region]= Macro_Kfold_RF_diffSet_ThreeClass(Label,Med_Feature{Itr,Region},Fold_Num,Med_Feature_Reg{Itr,Region});         
          
            RF_Full{Itr,Region} = RF_Full_Model;     RF_Reg{Itr,Region} =RF_Region;    
                       
       end

    end 
 
poolobj = gcp('nocreate');
delete(poolobj); 
display('PoolClosed')

toc 

save([DIR,'/Model_RF_ThreeClass_InRegion_100'],...
    'RF_Full','RF_Reg','Fold_Num','Feature','Feature_Reg','RandIndexNeg',...
    'Med_Feature','Med_Feature_Reg','RandIndexPos','RandIndexBaseline','Feat_Label','Feat_Label_Reg','-v7.3')

