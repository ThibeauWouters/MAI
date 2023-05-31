clear;
clf;
close all;
clc;

%set(groot, 'defaultAxesFontSize', 16)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robust regression with SVMs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Preparation
% Add SVM and LSSVM toolbox
addpath('/MATLAB Drive/SVM/svm/');
addpath('/MATLAB Drive/SVM/lssvm/');
type='f';

filename = "Data/weigh_results.csv";
header = {"wFun", "MSE"};
writecell(header, filename);

% Get data
X = (-6:0.2:6)';
Y = sinc(X) + 0.1.* rand(size(X));

Xtest = (-6:0.05:6)';
Ytest = sinc(Xtest);

% Add outliers
out = [15 17 19];
Y(out) = 0.7 + 0.3 * rand(size(out));
out    = [41 44 46];
Y(out) = 1.5 + 0.2 * rand(size(out));

%% Regular LS SVM
model = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');
costFun = 'crossvalidatelssvm';
model = tunelssvm(model,'simplex',costFun,{10,'mse';});

pred = simlssvm(model, Xtest);
mse_value = mean(mean((Ytest - pred).^2));

writecell({"None", mse_value}, filename, 'WriteMode', 'append');


figure;
plotlssvm(model);
hold on; plot(min(X):.1:max(X),sinc(min(X):.1:max(X)),'r-.');
mytitleText = sprintf('Standard regression (MSE: %.4f) ', mse_value);
title(mytitleText,'Interpreter','tex');

save_txt = 'Plots/outliers_normal.png';
exportgraphics(gca,save_txt, 'Resolution', 600);


%% Robust LS-SVM - Hampel and logistic
model = initlssvm (X, Y, 'f', [], [], 'RBF_kernel');
costFun = 'rcrossvalidatelssvm';
% wFun options: 'whuber', 'whampel', 'wlogistic' and 'wmyriad'.
wFun_list = {'whuber', 'wlogistic', 'whampel', 'wmyriad'};



for i=1:length(wFun_list)
    wFun = wFun_list{i};
    disp(wFun);

    model = tunelssvm(model,'simplex',costFun,{10,'mae'}, wFun);
    model = robustlssvm(model);

    % Get MSE
    pred = simlssvm(model, Xtest);
    mse_value = mean(mean((Ytest - pred).^2));

    writecell({wFun, mse_value}, filename, 'WriteMode', 'append');


    figure;
    plotlssvm(model);
    hold on; plot(min(X):.1:max(X),sinc(min(X):.1:max(X)),'r-.');

    mytitleText = sprintf('Robust regression: %s (MSE:%.4f) ', wFun, mse_value);
    title(mytitleText,'Interpreter','tex');
    
    save_txt = sprintf('Plots/outliers_robust_%s.png', wFun);
    exportgraphics(gca,save_txt, 'Resolution', 600);
end
