function [RF_Full_Model]= Macro_Kfold_RF_diffSetGeneralize(Label,Patient_ID,Feature,Fold_Num)


% ---------------Author : Maryam Bijanzadeh 11/24/2020 -------------


% decodes and generalizes cross 6 subjects using OFC, Insula and DCin 


%------------Train models based on this features for gain 

for K=Fold_Num
    
    fprintf('Leave One Out Patient # = %d \n',K)
    tic 
    
    Train = logical(Patient_ID~=K);  Test = logical(Patient_ID==K); 
    [RF_Full_Model(K)]=Train_Kfold_RandomForest_V2(Feature,Label,Train,Test,1) ;

  
    a= 1; 
    while size(RF_Full_Model(K).Xlog,1) < sum(Test) && a<5
         [RF_Full_Model(K)]=Train_Kfold_RandomForest_V2(Feature,Label,Train,Test,1) ;
         a = a+1; 
    end
   
 
    toc 
end



 