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

% Save data to external files
%writematrix(X, "sinc_X.txt");
%writematrix(Y, "sinc_Y.txt");

filename = 'sinc_results.csv';
header = {'Gamma','Sig2','MSE'};
writecell(header, filename);

% Separate into training and test data
Xtrain = X(1:2:end);
Ytrain = Y(1:2:end);
Xtest = X(2:2:end);
Ytest = Y(2:2:end);

% Define paramvals
gamlist = logspace(1, 6, 6);
sig2list = logspace(-2, 3, 6);

gamlist = [1000000];
sig2list = [1000];
type = 'function estimation';
for gam=gamlist
    for sig2=sig2list

        %% Random validation split
        [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

        % Make predictions
        Ypred = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);
        
        % Get MSE
        %mse = mean((Ypred - Ytest).^2);

        %% Use cross validation instead
        %mse = crossvalidate({Xtrain , Ytrain , 'f', gam , sig2 ,'RBF_kernel'}, 10,'mse');
        %results = {gam, sig2, mse};
        %writecell(results, filename, 'WriteMode', 'append');
    end
end

% % Plot
plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
 % Also plot the real values:
hold on; plot(min(X):.1:max(X),sinc(min(X):.1:max(X)),'r-.');

%%% Plot the LS SVM
plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
hold on; plot(min(X):.1:max(X),sinc(min(X):.1:max(X)),'r-.');
mytitleText = ['\gamma = ', num2str(gam,3),', \sigma^2 = ', num2str(sig2,3)];
title(mytitleText,'Interpreter','tex');

save_txt = sprintf('Plots/sinc_gam_%f_sig_%f.png', gam, sig2);
exportgraphics(gca,save_txt, 'Resolution', 600);