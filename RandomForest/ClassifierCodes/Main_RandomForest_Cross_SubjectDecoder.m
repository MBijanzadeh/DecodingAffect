


% this will look at the train cross subject decoders using each using
% features from 
% from OFC, Insula and DCin across 6
% patients and test on another patient , it runs it making 10 random
% dataset from each patient 
% generates Data for figure 5 A and D 

% also generates 100 datasets 


fileID = fopen('/tmp/DecoderParams_RF_Gen.txt','r'); 
params = textscan(fileID,'%s \n %s \n %s'); 


% Functions and subfunction should be added to the main path 

% DIR = Navigate to a directory where you have the fetaures : 
% and load the matfile containing features e.g.:
% Pos_Feature : frequency  x positve samples x channel number
% Pos_Feature_Reg : frequency  x positve samples x region number


rng(19840901);

Affect_Type = {cell2char(params{1}),cell2char(params{2})}  

cd(DIR) 
% 
load([DIR '/Pos_Data_7Patients_V2']) %loading all the data from subjects

poolobj = gcp('nocreate');
delete(poolobj);
G= parpool(5); 
Itr_Num = 10;


tic 
    if strcmp(Affect_Type{1},'Pos') && strcmp(Affect_Type{2},'Base')
        
         
        RF_Full = cell(Itr_Num,1); 
        RandIndexPos = cell(numel(Data),Itr_Num); RandIndexNeg = cell(numel(Data),Itr_Num); RandIndexBaseline = cell(numel(Data),Itr_Num); 
        Feature = cell(Itr_Num,1); Positive_Feature = cell(Itr_Num,1);  Neutral_Feature = cell(Itr_Num,1); 
        Label_All = cell(Itr_Num,1);  Label_Pos = cell(Itr_Num,1); Label_Neutral = cell(Itr_Num,1); 

        Fold_Num = numel(Data); 
        
        % generate the test and train dataset for each patient , this is
        % leaving one subject out. however the neutral class always has
        % larger sample number thus we will chose a subset of it 10 times
        % from all patients

        
        % build the datasets
        for Itr = 1: Itr_Num
           fprintf('forming Dataset # = %d \n',Itr)

            Positive_Feature{Itr} = []; Neutral_Feature{Itr} = [];Label_Pos{Itr} =[]; Label_Neutral{Itr} =[]; 
            Patient_id{Itr} = []; 
            
            for K = [1:3 5:7] % this one makes them balance for each patinet and then catenates them for each Itr_Num, 
                Num = min(size(Data(K).Pos_Feat_Reg,2),size(Data(K).Base_Feat_Reg,2)); 
                RandIndexPos{K,Itr} = sort(randperm(size(Data(K).Pos_Feat_Reg,2),Num));
                RandIndexBaseline{K,Itr} = sort(randperm(size(Data(K).Base_Feat_Reg,2),Num));
                Positive_Feature{Itr} = [Positive_Feature{Itr}; Data(K).Pos_Feat_Reg(1:15,RandIndexPos{K,Itr})' ]; % just includes the OFC, insula and Dcin
                Neutral_Feature{Itr} = [ Neutral_Feature{Itr}; Data(K).Base_Feat_Reg(1:15,RandIndexBaseline{K,Itr})' ];
                Label_Pos{Itr} = [Label_Pos{Itr}; ones(Num,1)*K];
                Label_Neutral{Itr} = [Label_Neutral{Itr}; ones(Num,1)*K];

            end
            
            Feature{Itr} = [Positive_Feature{Itr}; Neutral_Feature{Itr}]; 
            Label_All{Itr} = [ones(size(Positive_Feature{Itr},1),1); zeros(size(Positive_Feature{Itr},1),1)]; 
            Patient_id{Itr} = [Label_Pos{Itr}; Label_Neutral{Itr}]; 

        end 
      
        
        parfor Itr= 1:Itr_Num
                     
            [RF_Full_Model]= Macro_Kfold_RF_diffSetGeneralize(Label_All{Itr},Patient_id{Itr},Feature{Itr},[1:3 5:7]);         

            RF_Full{Itr} = RF_Full_Model;    
        end
        
      
    elseif strcmp(Affect_Type{1},'Neg') && strcmp(Affect_Type{2},'Base')
%         Patients with OFC, Insula and Dcin are K = 1,2, 5, 6 
        RF_Full = cell(Itr_Num,1); 
        RandIndexPos = cell(numel(Data),Itr_Num); RandIndexNeg = cell(numel(Data),Itr_Num); RandIndexBaseline = cell(numel(Data),Itr_Num); 
        Feature = cell(Itr_Num,1); Negative_Feature = cell(Itr_Num,1);  Neutral_Feature = cell(Itr_Num,1); 
        Label_All = cell(Itr_Num,1);  Label_Neg = cell(Itr_Num,1); Label_Neutral = cell(Itr_Num,1); 

        Fold_Num = 4; % This is technically patient number
        
        % generate the test and train dataset for each patient , this is
        % leaving one subject out. however the neutral class always has
        % larger sample number thus we will chose a subset of it 10 times
        % from all patients
%         build the feature sets


        for Itr = 1: Itr_Num
           fprintf('forming Dataset # = %d \n',Itr)

            Negative_Feature{Itr} = []; Neutral_Feature{Itr} = []; Label_Neg{Itr} =[]; Label_Neutral{Itr} =[]; 
            Patient_id{Itr} = []; 
            for K = [1 2 5 6] % this one makes them balance for each patinet and then catenates them for each Itr_Num
                Num = min(size(Data(K).Neg_Feat_Reg,2),size(Data(K).Base_Feat_Reg,2)); 
                RandIndexNeg{K,Itr} = sort(randperm(size(Data(K).Neg_Feat_Reg,2),Num));
                RandIndexBaseline{K,Itr} = sort(randperm(size(Data(K).Base_Feat_Reg,2),Num));
                Negative_Feature{Itr} = [Negative_Feature{Itr}; Data(K).Neg_Feat_Reg(1:15,RandIndexNeg{K,Itr})' ];
                Neutral_Feature{Itr} = [ Neutral_Feature{Itr}; Data(K).Base_Feat_Reg(1:15,RandIndexBaseline{K,Itr})' ];
                Label_Neg{Itr} = [Label_Neg{Itr}; ones(Num,1)*K];
                Label_Neutral{Itr} = [Label_Neutral{Itr}; ones(Num,1)*K];

            end
            
            Feature{Itr} = [Negative_Feature{Itr}; Neutral_Feature{Itr}]; 
            Label_All{Itr} = [ones(size(Negative_Feature{Itr},1),1); zeros(size(Negative_Feature{Itr},1),1)]; 
            Patient_id{Itr} = [Label_Neg{Itr}; Label_Neutral{Itr}]; 

        end 
      
        
        parfor Itr= 1:Itr_Num
                     
            [RF_Full_Model]= Macro_Kfold_RF_diffSetGeneralize(Label_All{Itr},Patient_id{Itr},Feature{Itr},[1,2,5,6]);         

            RF_Full{Itr} = RF_Full_Model;    
           
            
        end
        
   
   
    end
    
     poolobj = gcp('nocreate')
     delete(poolobj); 
     display('PoolClosed')


toc 
save([DIR,'/Model_RF_100Gen' Affect_Type{1} '_' Affect_Type{2}],...
    'RF_Full','Affect_Type','Fold_Num','Feature','RandIndexNeg','RandIndexPos','RandIndexBaseline','Label_All','Patient_id','Data','-v7.3')

