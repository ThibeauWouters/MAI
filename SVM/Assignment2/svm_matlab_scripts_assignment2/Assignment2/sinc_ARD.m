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

% Create the data
nb_points = 100;
Xtrain = 6.* rand(nb_points, 3) - 3;
Ytrain = sinc(Xtrain(:,1)) + 0.1.* randn(100,1);



%% ARD
[gam, sig2, ~] = tunelssvm({Xtrain,Ytrain,'f', [], [], 'RBF_kernel'}, "gridsearch", 'crossvalidatelssvm', {10, 'mse'}) ;
[selected,ranking] = bay_lssvmARD({Xtrain,Ytrain,'f',gam,sig2});

disp(selected);

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

%%% Plot the LS SVM
%plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
%hold on; plot(min(X):.1:max(X),sinc(min(X):.1:max(X)),'r-.');
%mytitleText = ['\gamma = ', num2str(gam,3),', \sigma^2 = ', num2str(sig2,3)];
%title(mytitleText,'Interpreter','tex');

%save_txt = sprintf('Plots/sinc_gam_%f_sig_%f.png', gam, sig2);
%exportgraphics(gca,save_txt, 'Resolution', 600);



