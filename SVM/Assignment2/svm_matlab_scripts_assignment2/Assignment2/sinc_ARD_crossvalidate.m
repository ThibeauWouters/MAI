clear;
clf;
close all;
clc;

%set(groot, 'defaultAxesFontSize', 16)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function estimation with SVMs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add SVM and LSSVM toolbox
addpath('/MATLAB Drive/SVM/svm/');
addpath('/MATLAB Drive/SVM/lssvm/');
type='f';

filename = 'sinc_ARD_CV.csv';
header = {'Subset','MSE'};
writecell(header, filename);

% Create the data
nb_points = 500;
Xtrain = 6.* rand(nb_points, 3) - 3;
Ytrain = sinc(Xtrain(:,1)) + 0.1.* randn(nb_points,1);

% Get subsets for training
Xtrain_1  = Xtrain(:, 1);
Xtrain_2  = Xtrain(:, 2);
Xtrain_3  = Xtrain(:, 3);
Xtrain_12 = Xtrain(:, [1, 2]);
Xtrain_13 = Xtrain(:, [1, 3]);
Xtrain_23 = Xtrain(:, [2, 3]);

train_sets = {Xtrain_12, Xtrain_13, Xtrain_23, Xtrain_1, Xtrain_2, Xtrain_3};
names = ["1, 2", "1, 3", "2, 3", "1", "2", "3"];

%% Original MSE
[gam, sig2, ~] = tunelssvm({Xtrain,Ytrain,'f', [], [], 'RBF_kernel'}, "gridsearch", 'crossvalidatelssvm', {10, 'mse'});
mse = crossvalidate({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},10,'mse');

results = {"All", mse};
writecell(results, filename, 'WriteMode', 'append');

for i=1:length(train_sets)
    set = train_sets{i};
    name = names(i);

    [gam, sig2, ~] = tunelssvm({set,Ytrain,'f', [], [], 'RBF_kernel'}, "gridsearch", 'crossvalidatelssvm', {10, 'mse'});
    mse = crossvalidate({set,Ytrain,'f',gam,sig2,'RBF_kernel'},10,'mse');
    results = {name, mse};
    writecell(results, filename, 'WriteMode', 'append');
end