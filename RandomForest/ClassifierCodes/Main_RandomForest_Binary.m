


%% ------------------------ Author: Maryam Bijanzadeh  ----------------

% trains both Random forest  both full feature set from all
% electrodes as well within reagion averages 

% also generates 100 datasets

% gets the paramter from a text file
fileID = fopen('/tmp/DecoderParams_RF_EN.txt','r'); 
params = textscan(fileID,'%s \n %s \n %s'); 

% Functions and subfunction should be added to the main path 

% DIR = Navigate to a directory where you have the fetaures : 
% and load the matfile containing features e.g.:
% Pos_Feature : frequency  x positve samples x channel number
% Pos_Feature_Reg : frequency  x positve samples x region number

rng(19840901);

Affect_Type = {cell2char(params{2}),cell2char(params{3})}  
cd(DIR) 
msg = sprintf('About to start running with EC:  %s and Affect_Type: %s and %s ', cell2char(params{1}),cell2char(params{2}),cell2char(params{3}));
msg
 

poolobj = gcp('nocreate');
delete(poolobj);
G= parpool(5); 

tic 
    if strcmp(Affect_Type{1},'Pos') && strcmp(Affect_Type{2},'Base')
        
         Num = min(Pos_Num,size(Power_Random,2)); 
         Label = [ones(Num,1); zeros(Num,1)];

           if Num<60
             Fold_Num = 5 ;

           else 
            Fold_Num = 10; 
           end
  
        [Partition,Fold_Num_New] = Partition_Consecutive_Time(Label,Fold_Num) ; 

         Itr_Num = floor(100/Fold_Num_New);
            
            RF_Full = cell(Itr_Num,1);  RF_Reg = cell(Itr_Num,1);  
            RandIndexPos = cell(1,Itr_Num); RandIndexNeg = cell(1,Itr_Num); RandIndexBaseline = cell(1,Itr_Num); RandIndexConv = cell(1,Itr_Num); 
            Feature = cell(Itr_Num,1);  Feature_Reg = cell(Itr_Num,1);
      
        parfor Itr= 1:Itr_Num
            fprintf('Dataset # = %d \n',Itr)

            RandIndexPos{Itr} = sort(randperm(size(Pos_Feature,2),Num)); % Pos Feature = 6 x samples x number of channels 
            RandIndexBaseline{Itr} = sort(randperm(size(Power_Random,2),Num));

            Feature{Itr}= [reshape(permute(Pos_Feature([1:4 6],RandIndexPos{Itr},:),[1 3 2]),[],numel(RandIndexPos{Itr})) ...
                 reshape(permute(Power_Random([1:4 6],RandIndexBaseline{Itr},:),[1 3 2]),[],numel(RandIndexBaseline{Itr}))]' ; 
            
    
            Feature_Reg{Itr}= [reshape(permute(Pos_Feature_Reg([1:4 6],RandIndexPos{Itr},:),[1 3 2]),[],numel(RandIndexPos{Itr})) ...
                 reshape(permute(Base_Feature_Reg([1:4 6],RandIndexBaseline{Itr},:),[1 3 2]),[],numel(RandIndexBaseline{Itr}))]' ;                      
              
            
            [RF_Full_Model, RF_Region]= Macro_Kfold_RF_diffSet(Label,Feature{Itr},Fold_Num,Feature_Reg{Itr});         

            RF_Full{Itr} = RF_Full_Model;     RF_Reg{Itr} =RF_Region;    
           
            
        end
        
        

    elseif strcmp(Affect_Type{1},'Neg') && strcmp(Affect_Type{2},'Base')
         Num = min(Neg_Num,size(Power_Random,2)); 
         Label = [ones(Num,1); zeros(Num,1)];
         
          if Num<60
            Fold_Num = 5 ;

          else 
            Fold_Num = 10; 
          end

        [Partition,Fold_Num_New] = Partition_Consecutive_Time(Label,Fold_Num) ; 

         Itr_Num = floor(100/Fold_Num_New) 
            
            EN_Full = cell(Itr_Num,1);  EN_Reg= cell(Itr_Num,1); 
            RF_Full = cell(Itr_Num,1);  RF_Reg = cell(Itr_Num,1);  
            RandIndexPos = cell(1,Itr_Num); RandIndexNeg = cell(1,Itr_Num); RandIndexBaseline = cell(1,Itr_Num); RandIndexConv = cell(1,Itr_Num); 
            Feature = cell(Itr_Num,1);  Feature_Reg = cell(Itr_Num,1);

         
         
       parfor Itr= 1:Itr_Num
           fprintf('Dataset # = %d \n',Itr)

            RandIndexNeg{Itr} = sort(randperm(size(Neg_Feature,2),Num));
            RandIndexBaseline{Itr} = sort(randperm(size(Power_Random,2),Num));
           
            Feature{Itr}= [reshape(permute(Neg_Feature([1:4 6],RandIndexNeg{Itr},:),[1 3 2]),[],numel(RandIndexNeg{Itr})) ...
                 reshape(permute(Power_Random([1:4 6],RandIndexBaseline{Itr},:),[1 3 2]),[],numel(RandIndexBaseline{Itr}))]' ; 
            
            Feature_Reg{Itr}= [reshape(permute(Neg_Feature_Reg([1:4 6],RandIndexNeg{Itr},:),[1 3 2]),[],numel(RandIndexNeg{Itr})) ...
                 reshape(permute(Base_Feature_Reg([1:4 6],RandIndexBaseline{Itr},:),[1 3 2]),[],numel(RandIndexBaseline{Itr}))]' ; 
    
         
           [EN_Model,RF_Full_Model, EN_Region, RF_Region]= Macro_Kfold_RF_ELNet_diffSet(Label,Feature{Itr},Fold_Num,Feature_Reg{Itr});         

            EN_Full{Itr} = EN_Model;   EN_Reg{Itr} = EN_Region; 
            RF_Full{Itr} = RF_Full_Model;     RF_Reg{Itr} = RF_Region;        
       
       end 
        
       
   
        
    elseif strcmp(Affect_Type{1},'Neg') && strcmp(Affect_Type{2},'Pos')
            Num = min(Pos_Num,Neg_Num);
            Label = [ones(Num,1); zeros(Num,1)];
            display('Khar')

           if Num<60
             Fold_Num = 5 ;

           else 
            Fold_Num = 10; 
           end
  
        [Partition,Fold_Num_New] = Partition_Consecutive_Time(Label,Fold_Num) ; 

         Itr_Num = floor(100/Fold_Num_New)
            
            EN_Full = cell(Itr_Num,1);  EN_Reg= cell(Itr_Num,1); 
            RF_Full = cell(Itr_Num,1);  RF_Reg = cell(Itr_Num,1);  
            RandIndexPos = cell(1,Itr_Num); RandIndexNeg = cell(1,Itr_Num); RandIndexBaseline = cell(1,Itr_Num); RandIndexConv = cell(1,Itr_Num); 
            Feature = cell(Itr_Num,1);  Feature_Reg = cell(Itr_Num,1);

        
        parfor Itr= 1:Itr_Num
            fprintf('Dataset # = %d \n',Itr)

            RandIndexNeg{Itr} = sort(randperm(size(Neg_Feature,2),Num)); % Pos Feature = 6 x samples x number of channels 
            RandIndexPos{Itr} = sort(randperm(size(Pos_Feature,2),Num));

            Feature{Itr}= [reshape(permute(Neg_Feature([1:4 6],RandIndexNeg{Itr},:),[1 3 2]),[],numel(RandIndexNeg{Itr})) ...
                 reshape(permute(Pos_Feature([1:4 6],RandIndexPos{Itr},:),[1 3 2]),[],numel(RandIndexPos{Itr}))]' ; 
                
            Feature_Reg{Itr}= [reshape(permute(Neg_Feature_Reg([1:4 6],RandIndexNeg{Itr},:),[1 3 2]),[],numel(RandIndexNeg{Itr})) ...
                 reshape(permute(Pos_Feature_Reg([1:4 6],RandIndexPos{Itr},:),[1 3 2]),[],numel(RandIndexPos{Itr}))]' ;                      
              
            
            [RF_Full_Model, RF_Region]= Macro_Kfold_RF_diffSet(Label,Feature{Itr},Fold_Num,Feature_Reg{Itr});         

            RF_Full{Itr} = RF_Full_Model;     RF_Reg{Itr} =RF_Region;    
           
            
        end

   
    end
    
     poolobj = gcp('nocreate');
     delete(poolobj); 
     display('PoolClosed')


toc 

save([DIR,'/Model_RF_100_' Affect_Type{1} '_' Affect_Type{2}],...
    'RF_Full','RF_Reg','Affect_Type','Fold_Num','Feature','Feature_Reg','RandIndexNeg','RandIndexSoc','RandIndexPos','RandIndexRest','RandIndexBaseline','RandIndexConv','Feat_Label','Feat_Label_Reg','-v7.3')

