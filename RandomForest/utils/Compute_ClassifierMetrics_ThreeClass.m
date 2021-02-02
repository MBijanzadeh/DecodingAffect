function [ConMat,Recall,FValue,Precision,Accuracy_Avg,PostProbs,Newlabels,NewLabels_Ind,Flag] = Compute_ClassifierMetrics_ThreeClass(Trained_Model,Feature,Label,Ind_Train,Ind_Test) 


%% ------------------------Author:  Maryam Bijanzadeh  15/03/2019 ----------------
% This function computes Accuracy, confusion matrix, fvalue & posterior
% probality, It also passes a Flag if the posterior function is a perfect step function.  

% Inputs; 
% Trained_Model = is the model structure from Matlab SVM output
% Features = sample x features 
% Label = samples x 1 , vector of labels 
% Ind_Train = logical vector, is the training set coming from the partition , i.e. Partition.training 
% Ind_test = logical vector, is the test set from the partition , i.e. Partition.test

    [Newlabels,PostProbs] = predict(Trained_Model,Feature(Ind_Test,:));  % this line will predict labels for test data set and their posterior probability 

    
    if iscell(Newlabels)
        
        Newlabels = str2num(cell2mat(Newlabels)); 
    end
      ConMat = confusionmat(Label(Ind_Test),Newlabels); % computes the confusion matrix  using matlab built in function, confusionmat sorts the labels so the 
      % the first one is zero, second one is 1 and third one is 2 
  
    Accuracy_Avg= sum(diag(ConMat))/sum(reshape(ConMat,[],1));  % average Accuracy across all classes 
    Recall = (diag(ConMat)./sum(ConMat,2))';
    Precision = (diag(ConMat)./sum(ConMat,1)')';    
    FValue = (2*Recall.*Precision)./(Recall+Precision); 
    
    
    NewLabels_Ind = zeros(size(Ind_Test,1),1);   
    NewLabels_Ind(Ind_Test)= Newlabels; 
    

%     if strcmp(CompactSVMModel.ScoreTransform(5:8),'step') % there is a warning that the fitted probablity is the true step function ! 
%         Flag = 1; 
%     else 
        Flag = 0; 
%     end
