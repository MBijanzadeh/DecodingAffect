

function [RF_FullModel]=Train_Kfold_RandomForest(Feature,Label,Train,Test,Rand_iter) 

%% ------------------------ Author: Maryam Bijanzadeh  ----------------

% 
%  make the model much smaller 

% Inputs:
% Feature and Label=  sample x features and sample x 1 
% Train = logical vector, is the training set coming from the partition , i.e. Partition.training 
% test = logical vector, is the test set from the partition , i.e. Partition.test
% Rand iter = scalar defining how many times the labels are shuffled for
% the random model 

% Outputs: 
% RF_FullModel = a structure containing the trained model paramters that has Accuracy, AUC, Xlog, Ylog, FValue,
% confusion matrix and etc. 
% RF_RandomModel = a structure that is similar as RF_FullModel but
% the labels are shuffled for Rand_iter times on the same trained model -->
% this is testing how the accuracy of the trained model would be if the
% labels are shuffled. 
 
PostProbs_Full = []; ConMat_Full = []; Accuracy_Full =[];FValue_RF_Full = []; 
PostProbs_Rand = cell(1,Rand_iter); ConMat_Rand = cell(1 ,Rand_iter); Accuracy_Rand = zeros(1,Rand_iter); FValue_RF_Rand =zeros(1,Rand_iter); 
Opt_Params = struct; 
% Xlog_Rand = cell(1 ,Rand_iter); Ylog_Rand = cell(1 ,Rand_iter); Tlog_Rand = cell(1 ,Rand_iter); AUClog_Rand = zeros(1,Rand_iter); 
% Xlog = []; Ylog = []; Tlog = []; AUClog=[]; 

    [RF_Model,bestOOBErr,bestHyperparameters]= GetOptimizedModel_RandomForest(Feature,Label,Train);
   
    [Accuracy_Full,ConMat_Full,FValue_RF_Full,PostProbs_Full,NewLabels,NewLabels_Ind,Flag] = Compute_ClassifierMetrics(RF_Model,Feature,Label,Train,Test);
    Feature_Imp = RF_Model.OOBPermutedPredictorDeltaError;
    
   
    % if the posterior probability is a step function retrain the model and
    % do this for 10 times at most 
    counterFlag = 1;
    while Flag && counterFlag<11
        
        [RF_Model,bestOOBErr,bestHyperparameters]= GetOptimizedModel_RandomForest(Feature,Label,Train);  
        [Accuracy_Full,ConMat_Full,FValue_RF_Full,PostProbs_Full,NewLabels,NewLabels_Ind,Flag] = Compute_ClassifierMetrics(RF_Model,Feature,Label,Train,Test);
        Feature_Imp = RF_Model.OOBPermutedPredictorDeltaError;
   
        counterFlag = counterFlag+1;
    end 
    
    [Xlog,Ylog,Tlog,AUClog] = perfcurve(Label(Test),PostProbs_Full(:,2),1); % Xlog and Ylog are useful for plotting the ROC curve, see the matlab perfcurve  
    
    
  Opt_Params.bestOOBErr = bestOOBErr;   Opt_Params.bestHyperparameters= bestHyperparameters; 
        
        
        

%%

RF_FullModel = struct; 
RF_FullModel.Accuracy = Accuracy_Full; RF_FullModel.ConMat= ConMat_Full;  RF_FullModel.FValue = FValue_RF_Full; RF_FullModel.Xlog= Xlog;
RF_FullModel.Ylog = Ylog; RF_FullModel.AUC = AUClog;  RF_FullModel.OptimParams = Opt_Params; RF_FullModel.Flag = Flag; RF_FullModel.NewLabel= NewLabels ; % the new labels comparable with abel(Test) 
RF_FullModel.FeatureImp =  Feature_Imp; 




