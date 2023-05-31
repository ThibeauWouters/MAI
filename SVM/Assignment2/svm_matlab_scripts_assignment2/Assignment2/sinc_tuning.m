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

% Create the data
X = (-3:0.01:3)';
Y = sinc(X) + 0.1.*randn(length(X),1);

%%% Save data to external files
%writematrix(X, "sinc_X.txt");
%writematrix(Y, "sinc_Y.txt");

% To initialize the file:
filename = 'sinc_tuning_results.csv';
header = {'Algorithm','Gamma','Sig2','MSE'};
writecell(header, filename);

% Separate into training and test data
Xtrain = X(1:2:end);
Ytrain = Y(1:2:end);
Xtest = X(2:2:end);
Ytest = Y(2:2:end);

% Set up the LS-SVM tuning
type = 'function estimation';
algorithm_list = ["gridsearch", "simplex"];
nb_repetitions = 100;

for a=1:length(algorithm_list)
    algorithm = algorithm_list(a);
    fprintf("Checking out %s", algorithm);
    for i=1:nb_repetitions
        [gam, sig2, cost] = tunelssvm({Xtrain,Ytrain,'function estimation', [], [], 'RBF_kernel'}, algorithm, 'crossvalidatelssvm', {10, 'mse'}) ;
        [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
        % Make predictions
        Ypred = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);
        
        % Get MSE
        mse = mean((Ypred - Ytest).^2);
        
        results = {algorithm, gam, sig2, mse};
        writecell(results, filename, 'WriteMode', 'append');
    end
end
disp("Done!");