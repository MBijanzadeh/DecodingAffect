function [Partition,Fold_Num] = Partition_Consecutive_Time_ThreeClass(Label,Fold_Num) 


%%--------------Author: Maryam Bijanzadeh 14/03/2019---------------------
% This function generates similar to Kfold but the test sets in time 
% %%--------------Author:  Maryam Bijanzadeh ---------------------
% This function generates stratified Kfold but the test sets in time 
% doesn't overlap with train test,like the frist 10 percent is selected 
% for the test in fold 1 and the remainder is training set
% there will be no validation set This will be used by Macro_Kfold_RF or
% any other classifier 

% It is totllay anticausal , here we don't care if the test set is
% preceeding the train set, we just care the samples are not randomly
% selected to have same features close in time. Although still the end of test set and
% the begining of train set might share a feature since they are back to
% back
% this is for 3 class


fold_size= floor( numel(Label)/Fold_Num); 
fold_size_EachLabel= round(fold_size/3); 

Label_One_Ind = find(Label>0.7 & Label<1.1); 
Label_Minus_Ind = find(Label>1.5); 
Label_Zero_Ind = find(Label<0.5); 

Partition = struct; 

% 
for K=1:Fold_Num   % 
    
   
   Partition.training{K} = zeros(numel(Label),1); % This is in the same format as matlab partition to make the usage easier     
   Partition.test{K} = zeros(numel(Label),1); 
 
   if K==Fold_Num %% If the number of labels is not devidable by the fold number the extra ones would be include in the last fold  
     
       if numel(Label_One_Ind(fold_size_EachLabel*(K-1)+1:end))<fold_size_EachLabel
           continue
       else
         Partition.test{K}(Label_One_Ind(fold_size_EachLabel*(K-1)+1:end)) = Label(Label_One_Ind(fold_size_EachLabel*(K-1)+1:end));  
         Partition.test{K}(Label_Minus_Ind(fold_size_EachLabel*(K-1)+1:end)) = ones(numel(Label_Minus_Ind(fold_size_EachLabel*(K-1)+1:end)),1);   
         Partition.test{K}(Label_Zero_Ind(fold_size_EachLabel*(K-1)+1:end)) = ones(numel(Label_Zero_Ind(fold_size_EachLabel*(K-1)+1:end)),1);   
       end
   else 
       
   Partition.test{K}(Label_One_Ind(fold_size_EachLabel*(K-1)+1:fold_size_EachLabel*K)) = Label(Label_One_Ind(fold_size_EachLabel*(K-1)+1:fold_size_EachLabel*K));   
   Partition.test{K}(Label_Minus_Ind(fold_size_EachLabel*(K-1)+1:fold_size_EachLabel*K)) = ones(numel(fold_size_EachLabel*(K-1)+1:fold_size_EachLabel*K),1);      
   Partition.test{K}(Label_Zero_Ind(fold_size_EachLabel*(K-1)+1:fold_size_EachLabel*K)) = ones(numel(fold_size_EachLabel*(K-1)+1:fold_size_EachLabel*K),1);   
    
   end
    
   Partition.training{K}(setdiff([1:numel(Label)],find(Partition.test{K}))) =ones((numel(Label)-numel(find(Partition.test{K}))),1); 

    
    
end 


K = Fold_Num; 
if numel(Label_One_Ind(fold_size_EachLabel*(K-1)+1:end))<fold_size_EachLabel
    Partition.training(K)=[]; 
    Partition.test(K)=[]; 
    Fold_Num = Fold_Num-1; 
end
    
    

