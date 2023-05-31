close all;
clear;
clc;

%% Automatic tuning

% Add SVM and LSSVM toolbox
addpath('/MATLAB Drive/SVM/svm/');
addpath('/MATLAB Drive/SVM/lssvm/');

% Load the dataset
load("Data/iris.mat");
% We fix gamma = 1 and also type
gam = 1;
type = 'classification';

fname = "Data/tunelssvm_results.csv";
header = {"Algorithm", "Gam", "Sig2", "Cost"};
writecell(header, fname);


%disp("Automatic tuning, simplex: ");
%algorithm = 'simplex';
%[tunedgam, tunedsig2, cost] = tunelssvm({Xtrain,Ytrain,'c', [], [], 'RBF_kernel'}, algorithm, 'crossvalidatelssvm', {10, 'misclass'}) ;
%fprintf("gam: %.2f , sig2: %.2f", gam, sig2);
%fprintf("cost: %.2f", cost);

%disp("Automatic tuning, simplex: ");
algorithms_list = ["gridsearch", "simplex"];
nb_repetitions=1;
for i=1:length(algorithms_list)
    algorithm = algorithms_list(i);
    for j=1:nb_repetitions
        [gam, sig2, cost] = tunelssvm({Xtrain,Ytrain,'c', [], [], 'RBF_kernel'}, algorithm, 'crossvalidatelssvm', {10, 'misclass'}) ;
        header = {algorithm, gam, sig2, cost};
        writecell(header, fname, "WriteMode", "append");
    end
end

[alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
Y_latent = latentlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xtest);
[area,se,thresholds,oneMinusSpec,Sens]=roc(Y_latent,Ytest);

%fprintf("gam: %.2f , sig2: %.2f", gam, sig2);
%fprintf("cost: %.2f", cost);

%disp("---");
%disp("Cost:");
%disp(cost);