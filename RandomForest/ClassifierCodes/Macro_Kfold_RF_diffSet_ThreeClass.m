function [RF_Full_Model, RF_Region]= Macro_Kfold_RF_diffSet_ThreeClass(Label,Feature,Fold_Num,Feature_Reg)


%% ------------------------ Author: Maryam Bijanzadeh  ----------------




[Partition,Fold_Num] = Partition_Consecutive_Time_ThreeClass(Label,Fold_Num); 



%------------Train models based on this features for gain 

for K=1:Fold_Num
    fprintf('fold # = %d \n',K)
    tic 
    
    [RF_Full_Model(K)]=Train_Kfold_RandomForest_ThreeClass(Feature,Label,logical(cell2mat(Partition.training(K))),logical(cell2mat(Partition.test(K))),1) ;

  
     a= 1; 
    while size(RF_Full_Model(K).Xlog,1) < sum(cell2mat(Partition.test(1))) && a<5
         [RF_Full_Model(K)]=Train_Kfold_RandomForest_ThreeClass(Feature,Label,logical(cell2mat(Partition.training(K))),logical(cell2mat(Partition.test(K))),1) ;
    a = a+1; 
    end
    
   

    if nargin> 3
        
        
        [RF_Region(K)]=Train_Kfold_RandomForest_ThreeClass(Feature_Reg,Label,logical(cell2mat(Partition.training(K))),logical(cell2mat(Partition.test(K))),1) ;
  
  
        a=1; 
        while  size(RF_Region(K).Xlog,1) < sum(cell2mat(Partition.test(1)))  && a<5
         [RF_Region(K)]=Train_Kfold_RandomForest_ThreeClass(Feature_Reg,Label,logical(cell2mat(Partition.training(K))),logical(cell2mat(Partition.test(K))),1) ;
          a= a+1; 
        end  
        
        
        
    else
        RF_Region = []; 
    end 
    
    toc 
end



 