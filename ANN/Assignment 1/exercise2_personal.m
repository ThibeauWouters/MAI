clear
clc
close all

%%%%%%%%%%%
%exercise2_personal
% A script for the solution of exercise 2, the personal regression problem.
%%%%%%%%%%%


% Load the datafile as a structure array
data = importdata('Data/Data_Problem1_regression.mat');

% Store the variables in the workspace
X1 = data.X1;
X2 = data.X2;
T1 = data.T1;
T2 = data.T2;
T3 = data.T3;
T4 = data.T4;
T5 = data.T5;

% Digits for the student number r0708518
d1 = 8;
d2 = 8;
d3 = 7;
d4 = 5;
d5 = 1;

% Build personal dataset
Tnew = (d1*T1 + d2*T2 + d3*T3 + d4*T4 + d5*T5)/(d1 + d2 + d3 + d4 + d5);
% Export the data so we can plot it in Python
my_data = [X1, X2, Tnew];
writematrix(my_data, "Data/my_data.csv");

% Sample train, test, and validation samples
n_points = 1000;
% Training data (will be split into 50% training data and 50% validation
% data
train_X1   = datasample(X1, 2*n_points, 'Replace', false);
train_X2   = datasample(X2, 2*n_points, 'Replace', false);
train_Tnew = datasample(Tnew, 2*n_points, 'Replace', false);

train_input  = [train_X1'; train_X2'];
train_target = train_Tnew';

% Test data
test_X1   = datasample(X1, n_points, 'Replace', false);
test_X2   = datasample(X2, n_points, 'Replace', false);
test_Tnew = datasample(Tnew, n_points, 'Replace', false);

test_input  = [test_X1'; test_X2'];
test_target = test_Tnew';

%%% Plot the original data points and mesh in 3D

% Create 3D plot of the training data
F = scatteredInterpolant(X1, X2, Tnew);

% Define a grid of points at which to evaluate the surface
[Xq,Yq] = meshgrid(linspace(0, 1, 100));

% Evaluate the surface at the grid points
Zq = F(Xq, Yq);

%figure;
%plot3(X1, X2, Tnew,'o');
% Plot the interpolated surface using mesh
%mesh(Xq, Yq, Zq);

%%% Code from Toledo
f = scatteredInterpolant(train_X1, train_X2, train_Tnew);

% create a regular grid and evaluate the interpolations on it
x1 = linspace(min(train_X1), max(train_X1), 1000);
x2 = linspace(min(train_X2), max(train_X2), 1000);
[X1_mesh, X2_mesh] = meshgrid(x1, x2);
Y_mesh = f(X1_mesh, X2_mesh);

% Plot a 3D mesh of the grid with its interpolated values
figure;
mesh(X1_mesh, X2_mesh, Y_mesh);
title('Distribution of training data');
xlabel('X1');
ylabel('X2');
zlabel('Y');

% Optionally add markers to get an idea of how your data is distributed
hold on;
plot3(train_X1, train_X2, train_Tnew,'.','MarkerSize',15)
hold off;

%%% Train neural network

% Go train a neural net:
hidden_layer_size = [100, 100, 100];
num_iterations = 5000;
lr = 0.1;
alg = "traingdx";
% Define the network
net = feedforwardnet(hidden_layer_size, alg);
% Configure to the example
net = configure(net, train_input, train_target);
% Save the number of epochs to train
net.trainParam.epochs = num_iterations;
net.trainParam.lr = lr;
% Split into validation and training set for early stopping
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.5;
net.divideParam.valRatio   = 0.5;
net.divideParam.testRatio  = 0;
% Save the number of validation checks before early stopping 
net.trainParam.max_fail = 10;
% Randomly initialize the weights
net = init(net);
% Set hidden layer activation functions to tansig
num_hidden_layers = numel(net.layers) - 1;
for i = 1:num_hidden_layers
    net.layers{i}.transferFcn = 'tansig';
end
%%% Choose whether we show training or not
net.trainParam.showWindow = 0;
% Train the network
[net, tr] = train(net, train_input, train_target);
disp("Best performance on validation set:");
disp(tr.best_vperf);

%%% Plot training procedure for insights
%plotperform(tr);

% Get predictions on the test set
predictions = net(test_input);
% Get differences as well
differences = test_target - predictions;

%%% Plot test set and predictions
% Evaluate interpolations
x1 = linspace(min(test_X1), max(test_X1), 1000);
x2 = linspace(min(test_X2), max(test_X2), 1000);
[X1_mesh, X2_mesh] = meshgrid(x1, x2);
% Get interpolant and evaluate it
f = scatteredInterpolant(test_X1, test_X2, test_target');
Y_mesh = f(X1_mesh, X2_mesh);

% Plot a 3D mesh of the grid with its interpolated values
figure;
mesh(X1_mesh, X2_mesh, Y_mesh);
title('Errors');
xlabel('X1');
ylabel('X2');
zlabel('Y');

% Optionally add markers to get an idea of how your data is distributed
hold on;
plot3(test_X1, test_X2, test_target,'.','MarkerSize',15)
plot3(test_X1, test_X2, predictions,'.','MarkerSize',15, 'Color', 'red')
hold off;

%%% Plot errors on the test set
% Get differences
predictions = net(test_input);
differences = test_target - predictions;
f = scatteredInterpolant(test_X1, test_X2, differences');

% Evaluate interpolations
x1 = linspace(min(test_X1), max(test_X1), 1000);
x2 = linspace(min(test_X2), max(test_X2), 1000);
[X1_mesh, X2_mesh] = meshgrid(x1, x2);
Y_mesh = f(X1_mesh, X2_mesh);

% Plot a 3D mesh of the grid with its interpolated values
figure;
mesh(X1_mesh, X2_mesh, Y_mesh);
title('Errors');
xlabel('X1');
ylabel('X2');
zlabel('Y');

% Optionally add markers to get an idea of how your data is distributed
hold on;
plot3(test_X1, test_X2, differences','.','MarkerSize',15)
hold off;

