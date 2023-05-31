clear;
clf;
close all;
clc;

%%%%%%%%
% Diabetes
%%%%%%%%

% Add SVM and LSSVM toolbox
addpath('/MATLAB Drive/SVM/svm/');
addpath('/MATLAB Drive/SVM/lssvm/');

% Load the dataset
load("Data/diabetes.mat");
% We fix gamma = 1 and also type
gam = 1;
type = 'classification';

