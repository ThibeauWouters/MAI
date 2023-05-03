clear;
clf;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exercise 1.3, tuning with a validation set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add SVM and LSSVM toolbox
addpath('/MATLAB Drive/SVM/svm/');
addpath('/MATLAB Drive/SVM/lssvm/');

% Load the dataset
load("Data/iris.mat");
type = 'classification';

% Define paramvals
gamlist = logspace(-3, 3, 7);
sig2list = logspace(-3, 3, 7);

%% Random split
perflist = zeros(length(gamlist), length(sig2list));
for i=1:length(gamlist)
    gam = gamlist(i);
    for j=1:length(sig2list)
        sig2 = sig2list(i);
        perf = rsplitvalidate({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},0.80,'misclass');
        perflist(i, j) = perf;
    end
end

%disp("Results for random split: ");
%disp(perflist);
writematrix(perflist, 'Data/ex1_tuning_randomsplit.csv');

%% Cross validation 
perflist = zeros(length(gamlist), length(sig2list));
for i=1:length(gamlist)
    gam = gamlist(i);
    for j=1:length(sig2list)
        sig2 = sig2list(i);
        perf = crossvalidate({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},10,'misclass');
        perflist(i, j) = perf;
    end
end

%disp("Results for cross validation: ");
%disp(perflist);
writematrix(perflist, 'Data/ex1_tuning_crossvalidation.csv');

%% Leave one out
perflist = zeros(length(gamlist), length(sig2list));
for i=1:length(gamlist)
    gam = gamlist(i);
    for j=1:length(sig2list)
        sig2 = sig2list(i);
        perf = leaveoneout({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},'misclass');
        perflist(i, j) = perf;
    end
end

%disp("Results for leave one out: ");
%disp(perflist);
writematrix(perflist, 'Data/ex1_tuning_leavoneout.csv');

%% Automatic tuning
%disp("Automatic tuning, simplex: ");
algorithm = 'simplex';
[tunedgam, tunedsig2, cost] = tunelssvm({Xtrain,Ytrain,'c', [], [], 'RBF_kernel'}, algorithm, 'crossvalidatelssvm', {10, 'misclass'}) ;
%fprintf("gam: %.2f , sig2: %.2f", gam, sig2);
%fprintf("cost: %.2f", cost);

%disp("Automatic tuning, simplex: ");
algorithm = 'gridsearch';
[tunedgam, tunedsig2, cost] = tunelssvm({Xtrain,Ytrain,'c', [], [], 'RBF_kernel'}, algorithm, 'crossvalidatelssvm', {10, 'misclass'}) ;
%fprintf("gam: %.2f , sig2: %.2f", gam, sig2);
%fprintf("cost: %.2f", cost);

%% ROC

% For not tuned
gam = 0.1;
sig2 = 10;
[alpha, b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
[Yest, Ylatent] = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xtest);
roc(Ylatent, Ytest);
hold on;

[alpha, b] = trainlssvm({Xtrain,Ytrain,'c',tunedgam,tunedsig2,'RBF_kernel'});
[Yest, Ylatent] = simlssvm ({Xtrain,Ytrain,'c',tunedgam,tunedsig2,'RBF_kernel'}, {alpha,b},Xtest);
roc(Ylatent, Ytest);
hold off;