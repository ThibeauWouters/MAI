clear;
clf;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exercise 1.3, Bayesian framework
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add SVM and LSSVM toolbox
addpath('/MATLAB Drive/SVM/svm/');
addpath('/MATLAB Drive/SVM/lssvm/');

% Load the dataset
load("Data/iris.mat");
type = 'classification';

tunedgam = 2.2193;
tunedsig2 = 0.091373;

bay_modoutClass({Xtrain,Ytrain,'c',tunedgam,tunedsig2}, 'figure');
title("$\gamma = 2.2193, \sigma^2 = 0.091373$", 'Interpreter', 'latex');
exportgraphics(gca,'Plots/ex1_bayesian1.png', 'Resolution', 600);
exportgraphics(gca,'Plots/ex1_bayesian1.eps');
%colorbar;

%% Different parameters
bay_modoutClass({Xtrain,Ytrain,'c',0.1,0.1}, 'figure');
title("$\gamma = 0.1, \sigma^2 = 0.1$", 'Interpreter', 'latex');
exportgraphics(gca,'Plots/ex1_bayesian2.png', 'Resolution', 600);
exportgraphics(gca,'Plots/ex1_bayesian2.eps');
%colorbar;

bay_modoutClass({Xtrain,Ytrain,'c',0.1,1}, 'figure');
title("$\gamma = 0.1, \sigma^2 = 1$", 'Interpreter', 'latex');
exportgraphics(gca,'Plots/ex1_bayesian3.png', 'Resolution', 600);
exportgraphics(gca,'Plots/ex1_bayesian3.eps');
colorbar;

