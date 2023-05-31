clc;
clear;
close all;

%%%%%%%%%%%%%%%%%%%%%
% Fixed size LS-SVM %
%%%%%%%%%%%%%%%%%%%%%

% See: Stephane Caru: ell 0 optimization

%data = load('breast_cancer_wisconsin_data.mat','-ascii'); function_type = 'c';
data = load('shuttle.dat','-ascii'); 
data = data(1:700,:);
% data = load('california.dat','-ascii'); function_type = 'f';
% addpath('../LSSVMlab')

X = data(:,1:end-1);
Y = data(:,end);

% binarize the labels for shuttle data (comment these lines for
% california!)
Y(Y == 1) = 1;
Y(Y ~= 1) = -1;

testX = [];
testY = [];
function_type = 'c';  

%%

%Parameter for input space selection
%Please type >> help fsoperations; to get more information  

k = 4;
% function_type = 'c'; %'c' - classification, 'f' - regression  
function_type = 'c';
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);

