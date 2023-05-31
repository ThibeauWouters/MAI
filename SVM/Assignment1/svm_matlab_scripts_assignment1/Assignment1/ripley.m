clear;
clf;
close all;
clc;

set(groot, 'defaultAxesFontSize', 16)

%%%%%%%%
% Ripley
%%%%%%%%

% Add SVM and LSSVM toolbox
addpath('/MATLAB Drive/SVM/svm/');
addpath('/MATLAB Drive/SVM/lssvm/');

% Load the dataset
dataset_name = "diabetes";
capital_dataset_name = "Diabetes";
load_name = sprintf('Data/%s.mat', dataset_name);
load(load_name);
if ismember(dataset_name, ["breast", "diabetes"])
   Xtrain = trainset;
   Xtest  = testset;
   Ytrain = labels_train;
   Ytest  = labels_test;
end
type = 'classification';
algorithm = 'simplex';
prep = 'preprocess';

%% RBF

[gam, sig2, RBF_cost] = tunelssvm({Xtrain,Ytrain,'c', [], [], 'RBF_kernel'}, algorithm, 'crossvalidatelssvm', {10, 'misclass'}) ;

% Train the LS SVM
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

% Get the ROC
%figure;
Y_latent = latentlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'},{alpha,b},Xtest);
[area,se,thresholds,oneMinusSpec,Sens]=roc(Y_latent,Ytest);
title_text = sprintf('%s ROC RBF (AUC: %.4f)', capital_dataset_name, area);
title(title_text);

save_txt = sprintf('Plots/ex1_%s_RBF_ROC.png', dataset_name);
exportgraphics(gca,save_txt, 'Resolution', 600);

disp("---");
disp("RBF: Area under the curve:");
disp(area);
disp("---");

% Plot the LS SVM
figure; 
plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel',prep},{alpha,b});
mytitleText = [' RBF: \gamma = ', num2str(gam,3),', \sigma^2 = ',  num2str(sig2,3)];
title(mytitleText,'Interpreter','tex');
legend off;

save_txt = sprintf('Plots/ex1_%s_RBF.png', dataset_name);
exportgraphics(gca,save_txt, 'Resolution', 600);

%% Linear

[gam, params, linear_cost] = tunelssvm({Xtrain,Ytrain,'c', [], [], 'lin_kernel'}, algorithm, 'crossvalidatelssvm', {10, 'misclass'}) ;

% Train the LS SVM
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'});

% Get the ROC
%figure;
Y_latent = latentlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'},{alpha,b},Xtest);
[area,se,thresholds,oneMinusSpec,Sens]=roc(Y_latent,Ytest);
title_text = sprintf('%s ROC linear (AUC: %.4f)', capital_dataset_name, area);
title(title_text,'Interpreter','tex');
legend off

save_txt = sprintf('Plots/ex1_%s_linear_ROC.png', dataset_name);
exportgraphics(gca,save_txt, 'Resolution', 600);

disp("---");
disp("Linear: Area under the curve:");
disp(area);
disp("---");

% Plot the LS SVM
figure; 
plotlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel',prep},{alpha,b});
legend off;
mytitleText = [' Linear: \gamma = ', num2str(gam,3)];
title(mytitleText,'Interpreter','tex');
legend off;

save_txt = sprintf('Plots/ex1_%s_linear.png', dataset_name);
exportgraphics(gca,save_txt, 'Resolution', 600);


%% Polynomial

[gam, params, poly_cost] = tunelssvm({Xtrain,Ytrain,'c', [], [], 'poly_kernel'}, algorithm, 'crossvalidatelssvm', {10, 'misclass'}) ;

% Train the LS SVM
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,params,'poly_kernel'});

% Get the ROC
%figure;
Y_latent = latentlssvm({Xtrain,Ytrain,type,gam,params,'poly_kernel'},{alpha,b},Xtest);
[area,se,thresholds,oneMinusSpec,Sens]=roc(Y_latent,Ytest);
title_text = sprintf('%s ROC poly (AUC: %.4f)', capital_dataset_name, area);
title(title_text);

save_txt = sprintf('Plots/ex1_%s_poly_ROC.png', dataset_name);
exportgraphics(gca,save_txt, 'Resolution', 600);

disp("---");
disp("Poly: Area under the curve:");
disp(area);
disp("---");

% Plot the LS SVM
figure; 
plotlssvm({Xtrain,Ytrain,type,gam,params,'poly_kernel',prep},{alpha,b});
legend off;
mytitleText = [' Poly: \gamma = ', num2str(gam,3),', t = ',  num2str(params(1),3),', d = ',  num2str(params(2),3)];
title(mytitleText,'Interpreter','tex');

save_txt = sprintf('Plots/ex1_%s_poly.png', dataset_name);
exportgraphics(gca,save_txt, 'Resolution', 600);
