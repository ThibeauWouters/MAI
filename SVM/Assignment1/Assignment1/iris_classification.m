clear;
clf;

%%%%%%%%%%%%%%
% Exercise 1.3
%%%%%%%%%%%%%%

% Add SVM and LSSVM toolbox
addpath('/MATLAB Drive/SVM/svm/');
addpath('/MATLAB Drive/SVM/lssvm/');

% Load the dataset
load("Data/iris.mat");
% We fix gamma = 1 and also type
gam = 1;
type = 'classification';

%% Polynomial kernel

disp("Testing polynomial kernels");

t = 1;
errlist=[];
degree_list = [1, 2, 3, 4];
for degree=degree_list
    % Train the model
    [alpha, b] = trainlssvm({Xtrain, Ytrain, type, gam, [t;degree],'poly_kernel'});
    % Simulate it
    Yht = simlssvm({Xtrain, Ytrain, type, gam, [t;degree], 'poly_kernel', 'preprocess'}, {alpha, b}, Xtest);
    % Plot it
    %figure; 
    %plotlssvm({Xtrain, Ytrain, type, gam, [t;degree], 'poly_kernel', 'preprocess'}, {alpha,b});
    
    err = sum(Yht~=Ytest); errlist=[errlist; err];
    fprintf('\n Polynomial, degree %d on test: #misclass = %d, error rate = %.2f%% \n', degree, err, err/length(Ytest)*100)
end

disp("Testing polynomial kernels");

t = 1;
errlist=[];
acclist = []
degreelist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
for degree=degreelist
    % Train the model
    [alpha, b] = trainlssvm({Xtrain, Ytrain, type, gam, [t;degree],'poly_kernel'});
    % Simulate it
    Yht = simlssvm({Xtrain, Ytrain, type, gam, [t;degree], 'poly_kernel', 'preprocess'}, {alpha, b}, Xtest);
    % Plot it
    %figure; 
    %plotlssvm({Xtrain, Ytrain, type, gam, [t;degree], 'poly_kernel', 'preprocess'}, {alpha,b});
    
    err = sum(Yht~=Ytest); 
    acc = 1 - err/length(Ytest);
    errlist=[errlist; err];
    acclist=[acclist; acc];
    %fprintf('\n Polynomial, degree %d on test: #misclass = %d, error rate = %.2f%% \n', degree, err, err/length(Ytest)*100)
end

disp("Polynomial kernel analysis:");
disp("- degrees:");
disp(degreelist);
disp("- accuracies:");
disp(acclist');

%% RBF kernel

sig2list=[0.01, 0.1, 1, 5, 10, 15, 20, 25, 30, 35];
errlist=[];
acclist=[];
for sig2=sig2list
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    %plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
    
    err = sum(Yht~=Ytest); 
    acc = 1 - err/length(Ytest);
    errlist=[errlist; err];
    acclist=[acclist; acc];
    %fprintf('\n RBF with %.2f on test: #misclass = %d, error rate = %.2f%% \n', sig2, err, err/length(Ytest)*100)      
end

disp("RBF kernel analysis:");
disp("- sig2:");
disp(sig2list);
disp("- accuracies:");
disp(acclist');

%
% make a plot of the misclassification rate wrt. sig2
%
figure;
plot(log(sig2list), acclist, '*-'), 
xlabel('log(sig2)'), ylabel('Accuracy'),


%% Check gamma parameter

sig2 = 1;
gamlist = [0.001, 0.01, 0.1, 1, 10];
gamlist = 0.01:0.01:0.1
errlist=[];
acclist=[];
for gam=gamlist
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    
    %plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
    %pause;

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
    
    err = sum(Yht~=Ytest); 
    acc = 1 - err/length(Ytest);
    errlist=[errlist; err];
    acclist=[acclist; acc];
    %fprintf('\n RBF with %.2f on test: #misclass = %d, error rate = %.2f%% \n', sig2, err, err/length(Ytest)*100)      
end

disp("Gamma parameter analysis:");
disp("- sig2:");
disp(gamlist);
disp("- accuracies:");
disp(acclist');

%
% make a plot of the misclassification rate wrt. sig2
%
figure;
plot(log(gamlist), acclist, '*-'), 
xlabel('log(gamma)'), ylabel('Accuracy'),
