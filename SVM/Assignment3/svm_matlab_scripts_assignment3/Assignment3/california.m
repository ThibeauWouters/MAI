clc;
clear;
close all;

%%%%%%%%%%%%%%%%%%%%%
% Fixed size LS-SVM %
%%%%%%%%%%%%%%%%%%%%%

all_data = load('california.dat','-ascii'); 
function_type = 'f';
% addpath('../LSSVMlab')
N = 5000;
data = all_data(1:N,:);

X = data(:,1:end-1);
Y = data(:,end);

%Parameter for input space selection
%Please type >> help fsoperations; to get more information  

k = 4;
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'}; % 
window = [15,20,25];

[e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,[],[]);