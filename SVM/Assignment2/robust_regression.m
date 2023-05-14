clear;
clf;
close all;
clc;

%set(groot, 'defaultAxesFontSize', 16)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robust regression with SVMs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Preparation
% Add SVM and LSSVM toolbox
addpath('/MATLAB Drive/SVM/svm/');
addpath('/MATLAB Drive/SVM/lssvm/');
type='f';

% Get data
X = (-6:0.2:6)';
Y = sinc(X) + 0.1.* rand(size(X));

% Add outliers
out = [15 17 19];
Y(out) = 0.7 + 0.3 * rand(size(out));
out    = [41 44 46];
Y(out) = 1.5 + 0.2 * rand(size(out));

%% Regular LS SVM
model = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');
costFun = 'crossvalidatelssvm';
model = tunelssvm(model,'simplex',costFun,{10,'mse';});
figure;
plotlssvm(model);
hold on; plot(min(X):.1:max(X),sinc(min(X):.1:max(X)),'r-.');


%% Robust LS-SVM
model = initlssvm (X, Y, 'f', [], [], 'RBF_kernel');
costFun = 'rcrossvalidatelssvm';
% wFun options: 'whuber', 'whampel', 'wlogistic' and 'wmyriad'.
wFun = 'whuber';
model = tunelssvm(model,'simplex',costFun,{10,'mae'}, wFun);
model = robustlssvm(model);
figure;
plotlssvm(model);
