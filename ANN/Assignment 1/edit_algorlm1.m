clear
clc
close all

%%%%%%%%%%%
%algorlm.m
% A script comparing performance of 'trainlm' and 'trainbfg'
% trainbfg - BFGS (quasi Newton)
% trainlm - Levenberg - Marquardt
%%%%%%%%%%%

% Configuration:
alg1 = 'trainlm';% First training algorithm to use
alg2 = 'trainbfg';% Second training algorithm to use
H = 50;% Number of neurons in the hidden layer
epochs = [1000];% Number of epochs to train

% Generate training data
dx = 0.05;% Decrease this value to increase the number of data points
x=0:dx:3*pi;y=sin(x.^2);
sigma=0.2;% Standard deviation of added noise
yn=y+sigma*randn(size(y));% Add gaussian noise
t=y;% Targets. Change to yn to train on noisy data

% Define the network
net1 = feedforwardnet(H, alg1);
% Configure to the example
net1 = configure(net1, x, t);
% Use only training data
net1.divideFcn = 'dividetrain';
% Randomly initialize the weights
net1 = init(net1);

% Repeat for second network
net2 = feedforwardnet(H, alg2);
net2 = configure(net2, x, t);
net2.divideFcn = 'dividetrain';

% Set the same weights and biases for the networks 
net2.iw{1,1} = net1.iw{1,1};
net2.lw{2,1} = net1.lw{2,1};
net2.b{1}    = net1.b{1};
net2.b{2}    = net1.b{2};

% Don't show the training progress for net
net1.trainParam.showWindow = 0;
net2.trainParam.showWindow = 0;

% Training and simulation

% Set the number of epochs
net1.trainParam.epochs = epochs(1);  
net2.trainParam.epochs = epochs(1);
% Train the networks
disp("Training network...");
net1 = train(net1, x, t);
disp("Done!");
disp("Training network...");
net2 = train(net2, x, t);
disp("Done!");
% Simulate the networks with the input vector x
pred1 = sim(net1,x);
pred2 = sim(net2,x);

% Create plots
figure
% Plot the sine function and the output of the networks
plot(x, t, 'bx', x, pred1, 'r', x, pred2, 'g'); 
title([num2str(epochs(1)),' epochs']);
legend('target', alg1, alg2, 'Location', 'north');
% Perform a linear regression analysis and plot the result
%[m1, b1, r1] = postreg(pred1, y);
%[m2, b2, r2] = postreg(pred2, y);
%r1
%r2
