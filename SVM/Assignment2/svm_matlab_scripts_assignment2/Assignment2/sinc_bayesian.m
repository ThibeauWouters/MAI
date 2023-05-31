clear;
clf;
close all;
clc;

set(groot, 'defaultAxesFontSize', 16)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function estimation with SVMs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add SVM and LSSVM toolbox
addpath('/MATLAB Drive/SVM/svm/');
%addpath('/MATLAB Drive/SVM/lssvm/');

% Create the data
X = (-3:0.01:3)';
Y = sinc(X) + 0.1.*randn(length(X),1);

% Separate into training and test data
Xtrain = X(1:2:end);
Ytrain = Y(1:2:end);
Xtest = X(2:2:end);
Ytest = Y(2:2:end);

filename = 'sinc_Bayes_results.csv';
header = {'Gamma','Sig2','MSE', 'Nb repetitions'};
writecell(header, filename);

% Use Bayesian method to train
N = 5;
nb_Bayes_repetitions_list = linspace(1,N,N);
nb_Bayes_repetitions_list = [10];
disp(nb_Bayes_repetitions_list);

tic;
for a=1:length(nb_Bayes_repetitions_list)
    nb_Bayes_repetitions = nb_Bayes_repetitions_list(a);
    disp("Nb repetitions:");
    disp("nb_Bayes_repetitions");
    sig2 = 0.4;
    gam = 10;
    for i=1:nb_Bayes_repetitions
        crit_L1 = bay_lssvm({Xtrain,Ytrain, 'f', gam, sig2}, 1);
        crit_L2 = bay_lssvm({Xtrain,Ytrain, 'f', gam, sig2}, 2);
        crit_L3 = bay_lssvm({Xtrain,Ytrain, 'f', gam, sig2}, 3);
    
        [~, alpha ,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},1);
        [~, gam]      = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},2);
        [~, sig2 ]    = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},3);
    end
    
    % Get errorbars
    %figure;
    sig2e = bay_errorbar({Xtrain,Ytrain,'f',gam,sig2}, 'figure');
    mytitleText = ['\gamma = ', num2str(gam,3),', \sigma^2 = ', num2str(sig2,3)];
    title(mytitleText,'Interpreter','tex');
    save_txt = 'Plots/sinc_bayes_errorbars.png';
    exportgraphics(gca,save_txt, 'Resolution', 600);
    
    % Make predictions
    %[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
    Ypred = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);
    % Get MSE
    mse = mean((Ypred - Ytest).^2);
    disp(mse);

    % Write the results
    results = {gam, sig2, mse, nb_Bayes_repetitions};
    writecell(results, filename, 'WriteMode', 'append');
end
elapsed = toc;
disp("Elapsed time:");
disp(elapsed);

%%% Plot
%figure;
%plotlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
%hold on; plot(min(X):.1:max(X),sinc(min(X):.1:max(X)),'r-.');
%mytitleText = ['\gamma = ', num2str(gam,3),', \sigma^2 = ', num2str(sig2,3)];
%title(mytitleText,'Interpreter','tex');

%save_txt = sprintf('Plots/sinc_bayes_gam_%f_sig_%f.png', gam, sig2);
%exportgraphics(gca,save_txt, 'Resolution', 600);
