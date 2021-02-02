function [RF_Full_Model, RF_PCA]= RandomModel_Kfold_RF_PCA(Label,Feature,Fold_Num,Feature_Reg)


%% ------------------------ Author: Maryam Bijanzadeh  ----------------



    [Partition,Fold_Num] = Partition_Consecutive_Time(Label,Fold_Num); 
    Shuff_Label = zeros(numel(Label),Fold_Num);
    


% within each fold that has balanced labels we will suffle tlabels for both
% train and test
for K=1:Fold_Num
    fprintf('fold # = %d \n',K)
    tic 
    
    TestLabels = Label(logical(cell2mat(Partition.test(K)))); 
    Shuff_Test = TestLabels(randperm(size(TestLabels,1)));           
    Shuff_Label(logical(cell2mat(Partition.test(K))),K)= Shuff_Test; 

    TrainLabels = Label(logical(cell2mat(Partition.training(K)))); 
    Shuff_Train = TrainLabels(randperm(size(TrainLabels,1))); 
    Shuff_Label(logical(cell2mat(Partition.training(K))),K)= Shuff_Train; 
        
    %-------------- Full Model with all features ---------------------------
    
    [RF_Full_Model(K)] = Train_Kfold_RandomForest(Feature,Shuff_Label(:,K),logical(cell2mat(Partition.training(K))),logical(cell2mat(Partition.test(K))),1) ;

  
     a= 1; 
    while size(RF_Full_Model(K).Xlog,1) < sum(cell2mat(Partition.test(1))) && a<5
         [RF_Full_Model(K)]=Train_Kfold_RandomForest(Feature,Shuff_Label(:,K),logical(cell2mat(Partition.training(K))),logical(cell2mat(Partition.test(K))),1) ;
    a = a+1; 
    end
    
   
          %--------------  Feature_Reg   ---------------------------     

   
    if nargin> 3
        
          
          [RF_PCA(K)]=Train_Kfold_RandomForest(Feature_Reg,Shuff_Label(:,K),logical(cell2mat(Partition.training(K))),logical(cell2mat(Partition.test(K))),1) ;
  
  
        a=1; 
        while  size(RF_PCA(K).Xlog,1) < sum(cell2mat(Partition.test(1)))  && a<5
         [RF_PCA(K)]=Train_Kfold_RandomForest(Feature_Reg,Shuff_Label(:,K),logical(cell2mat(Partition.training(K))),logical(cell2mat(Partition.test(K))),1) ;
          a= a+1; 
        end  
        
        
        
    end       
          

    toc 
end



 