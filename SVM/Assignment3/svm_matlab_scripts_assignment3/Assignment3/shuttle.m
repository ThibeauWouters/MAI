clc;
clear;
close all;

%%%%%%%%%%%%%%%%%%%%%
% Fixed size LS-SVM %
%%%%%%%%%%%%%%%%%%%%%

%data = load('breast_cancer_wisconsin_data.mat','-ascii'); function_type = 'c';
all_data = load('shuttle.dat','-ascii'); 

% To limit nb of data?


%M = 1000;
N = 5000;
data = all_data(1:N,:);
size(data)
tic;

X = data(:,1:end-1);
Y = data(:,end);

% binarize the labels for shuttle data (comment these lines for
% california!)
Y(Y == 1) = 1;
Y(Y ~= 1) = -1;

%testdata = all_data(N+1:M,:);
%testX = testdata(:,1:end-1);
%testY = testdata(:,end);
function_type = 'c';  

%% Train model

k = 4;
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [10, 15, 20];

[e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,[],[]);

time_taken = toc;
disp("Time taken:");
disp(time_taken );