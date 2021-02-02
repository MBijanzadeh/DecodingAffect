


%% ------------------------ Author: Maryam Bijanzadeh  ----------------

% Trains 3 class Random forest for Neg-pos and netural with 100 datasets

fileID = fopen('/tmp/DecoderParams_RF_ThreeClass.txt','r'); 
params = textscan(fileID,'%s'); 


% Functions and subfunction should be added to the main path 

% DIR = Navigate to a directory where you have the fetaures : 
% and load the matfile containing features e.g.:
% Pos_Feature : frequency  x positve samples x channel number
% Pos_Feature_Reg : frequency  x positve samples x region number

rng(19840901);

Affect_Type = {cell2char(params{2}),cell2char(params{3})}  
cd(DIR) 

msg = sprintf('About to start running with EC:  %s ', cell2char(params{1}))

cd(DIR)
 
rng(19850623);
% 
 poolobj = gcp('nocreate');
 delete(poolobj);
 G= parpool(5); 

tic 
%  This selects ITr_Num different random samples from baselines 
   
         Num = min([Pos_Num,Neg_Num,size(Power_Random,2)]); 
         Label = [zeros(Num,1); ones(Num,1);ones(Num,1)*2 ]; %  neutral = 0 ,Pos labels = 1 , Neg Labels =2 
         
         
       if Num<60
         Fold_Num = 5 ;

       else 
        Fold_Num = 10; 
       end
  
        [Partition,Fold_Num_New] = Partition_Consecutive_Time_ThreeClass(Label,Fold_Num) ; 

         Itr_Num = floor(100/Fold_Num_New)
            
            RF_Full = cell(Itr_Num,1);  RF_Reg = cell(Itr_Num,1);  
            RandIndexPos = cell(1,Itr_Num); RandIndexNeg = cell(1,Itr_Num); RandIndexBaseline = cell(1,Itr_Num); 
            Feature = cell(Itr_Num,1);  Feature_Reg = cell(Itr_Num,1);
  
       parfor Itr= 1:Itr_Num 
            fprintf('Dataset # = %d \n',Itr)

            RandIndexPos{Itr} = sort(randperm(size(Pos_Feature,2),Num)); % Pos Feature = 6 x samples x number of channels 
            RandIndexBaseline{Itr} = sort(randperm(size(Power_Random,2),Num));
            RandIndexNeg{Itr} = sort(randperm(size(Neg_Feature,2),Num));
            
            Feature{Itr}= [reshape(permute(Power_Random([1:4 6],RandIndexBaseline{Itr},:),[1 3 2]),[],numel(RandIndexBaseline{Itr}))...
                           reshape(permute(Pos_Feature([1:4 6],RandIndexPos{Itr},:),[1 3 2]),[],numel(RandIndexPos{Itr})) ...
                           reshape(permute(Neg_Feature([1:4 6],RandIndexNeg{Itr},:),[1 3 2]),[],numel(RandIndexNeg{Itr}))]' ;                                
                       
                        
            Feature_Reg{Itr}= [ reshape(permute(Base_Feature_Reg([1:4 6],RandIndexBaseline{Itr},:),[1 3 2]),[],numel(RandIndexBaseline{Itr}))...
                                reshape(permute(Pos_Feature_Reg([1:4 6],RandIndexPos{Itr},:),[1 3 2]),[],numel(RandIndexPos{Itr})) ...
                                reshape(permute(Neg_Feature_Reg([1:4 6],RandIndexNeg{Itr},:),[1 3 2]),[],numel(RandIndexNeg{Itr})) ]' ;                  
                          
             
               
            [RF_Full_Model, RF_Region]= Macro_Kfold_RF_diffSet_ThreeClass(Label,Feature{Itr},Fold_Num_New,Feature_Reg{Itr});         

          
            RF_Full{Itr} = RF_Full_Model;     RF_Reg{Itr} =RF_Region;    
           
            
        end


poolobj = gcp('nocreate');
delete(poolobj); 
display('PoolClosed')


toc 

save([DIR,'/Model_RF_ThreeClass_100'],...
    'RF_Full','RF_Reg','Fold_Num','Feature','Feature_Reg','RandIndexNeg','RandIndexPos','RandIndexBaseline','Feat_Label','Feat_Label_Reg','-v7.3')

