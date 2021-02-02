

function [RF_FullModel]=Train_Kfold_RandomForest_ThreeClass(Feature,Label,Train,Test,Rand_iter) 

%% ------------------------ Author: Maryam Bijanzadeh  ----------------

% Similar as in Train_Kfold_RandomForest 
PostProbs_Full = []; ConMat_Full = []; Accuracy_Avg_Full =[];Recall_Full = []; FValue_Full =[]; Precision_Full=[]; 
PostProbs_Rand = cell(1,Rand_iter); ConMat_Rand = cell(1 ,Rand_iter); Accuracy_Rand = zeros(1,Rand_iter); FValue_RF_Rand =zeros(1,Rand_iter); 
Opt_Params = struct; 
   
[RF_Model,bestOOBErr,bestHyperparameters]= GetOptimizedModel_RandomForest(Feature,Label,Train);  
[ConMat_Full,Recall_Full, FValue_Full, Precision_Full,Accuracy_Avg_Full,PostProbs_Full,NewLabels,NewLabels_Ind,Flag] =  Compute_ClassifierMetrics_ThreeClass(RF_Model,Feature,Label,Train,Test);
Feature_Imp = RF_Model.OOBPermutedPredictorDeltaError;

   
    % if the posterior probability is a step function retrain the model and
    % do this for 10 times at most 
    counterFlag = 1;
    while Flag && counterFlag<11
        
        [RF_Model,bestOOBErr,bestHyperparameters]= GetOptimizedModel_RandomForest(Feature,Label,Train);  
        [ConMat_Full,Recall_Full, FValue_Full, Precision_Full,Accuracy_Avg_Full,PostProbs_Full,NewLabels,NewLabels_Ind,Flag] =  Compute_ClassifierMetrics_ThreeClass(RF_Model,Feature,Label,Train,Test);
        Feature_Imp = RF_Model.OOBPermutedPredictorDeltaError;
   
        counterFlag = counterFlag+1;
    end 
    
    [Xlog,Ylog,Tlog,AUClog] = perfcurve(Label(Test),PostProbs_Full(:,2),1); % Xlog and Ylog are useful for plotting the ROC curve, see the matlab perfcurve  
    
    
  Opt_Params.bestOOBErr = bestOOBErr;   Opt_Params.bestHyperparameters= bestHyperparameters; 
        
        
        

%%

RF_FullModel = struct; 
RF_FullModel.Accuracy = Accuracy_Avg_Full; RF_FullModel.ConMat= ConMat_Full;  RF_FullModel.Recall = Recall_Full; 
RF_FullModel.Precision = Precision_Full; RF_FullModel.FValue = FValue_Full;
RF_FullModel.Xlog= Xlog;
RF_FullModel.Ylog = Ylog; RF_FullModel.AUC = AUClog;  RF_FullModel.OptimParams = Opt_Params; RF_FullModel.Flag = Flag; RF_FullModel.NewLabel= NewLabels ; % the new labels comparable with abel(Test) 


RF_FullModel.FeatureImp =  Feature_Imp; 





