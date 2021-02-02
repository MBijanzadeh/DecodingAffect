

function [RF_Model,bestOOBErr,bestHyperparameters]= GetOptimizedModel_RandomForest(Feature,Label,Ind_Train) 

%% ------------------------ Author: Maryam Bijanzadeh  ----------------

% This function uses matlab built in treebagger functions to find the best parameters for training Random forest 
% there is no need for cross validation for paramter tuning in random
% forest bc OOB error takes care of it 
% Inputs : 
% Feature = samples x feautures 
% Label = samples by 1:  0s and 1s -> binary classifier 
% 

% uses this guide : https://www.mathworks.com/help/stats/tune-random-forest-using-quantile-error-and-bayesian-optimization.html


Optimized_Params= struct; 
%% 

Label1 = Label(Ind_Train);  
Feature_Opt1 = Feature(Ind_Train,:); 




maxMinLS = 20;
minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');
if size(Feature_Opt1,2) ==2 % In case of PCA 
    numPTS = optimizableVariable('numPTS',[1,2],'Type','integer');
else 
    numPTS = optimizableVariable('numPTS',[1,size(Feature_Opt1,2)-1],'Type','integer');
end
hyperparametersRF = [minLS; numPTS];



results = bayesopt(@(params)oobErrRF(params,Feature_Opt1,Label1),hyperparametersRF,...
    'AcquisitionFunctionName','expected-improvement-plus','Verbose',0,'PlotFcn',[]); 

bestOOBErr = results.MinObjective;
bestHyperparameters = results.XAtMinObjective;

RF_Model = TreeBagger(300,Feature_Opt1,Label1,...
    'OOBPrediction','on','Method','classification','Surrogate','On','OOBPredictorImportance','On',...
    'MinLeafSize',bestHyperparameters.minLS,...
    'NumPredictorstoSample',bestHyperparameters.numPTS);






%% 
function oobErr = oobErrRF(params,X,Y)
%oobErrRF Trains random forest and estimates out-of-bag quantile error
%   oobErr trains a random forest of 300 regression trees using the
%   predictor data in X and the parameter specification in params, and then
%   returns the out-of-bag quantile error based on the median. X is a table
%   and params is an array of OptimizableVariable objects corresponding to
%   the minimum leaf size and number of predictors to sample at each node.
randomForest = TreeBagger(300,X,Y,...
    'OOBPrediction','on','Method','classification','Surrogate','On','MinLeafSize',params.minLS,'OOBPredictorImportance','On',...
    'NumPredictorstoSample',params.numPTS);
oobErr = oobError(randomForest,'Mode','ensemble');
end


end

