clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%
% Session 1 exercise 2 %
%%%%%%%%%%%%%%%%%%%%%%%%

%% Read the data
data = load('Data_Problem1_regression.mat');
% Save into respective variables
X1 = data.X1;
X2 = data.X2;
T1 = data.T1;
T2 = data.T2;
T3 = data.T3;
T4 = data.T4;
T5 = data.T5;

% Get personalized dataset
Tnew = (8*T1 + 7*T2 + 7*T3 + 5*T4 + 1*T5)/(8 + 7 +  + 5 + 1);

%% Get samples -- old version
% For training
%X1_train = randsample(X1, 1000);
%X2_train = randsample(X2, 1000);
%T_train  = transpose(randsample(Tnew, 1000));

% For testing
%X1_test = randsample(X1, 1000);
%X2_test = randsample(X2, 1000);
%T_test  = transpose(randsample(Tnew, 1000));

% For validation
%X1_validate = randsample(X1, 1000);
%X2_validate = randsample(X2, 1000);
%T_validate  = transpose(randsample(Tnew, 1000));

% Get (X1, X2) as input for the perceptron
%X_train     = transpose([X1_train, X2_train]);
%X_test      = transpose([X1_test, X2_test]);
%X_validate  = transpose([X1_validate, X2_validate]);

%% Get samples -- new version

% Get individual samples
X1 = randsample(X1, 3000);
X2 = randsample(X2, 3000);
T  = transpose(randsample(Tnew, 3000));

% Get (X1, X2) as input for the perceptron
X = transpose([X1, X2]);


%% Configuration of the algorithms, networks and epochs:
H = 100; % Number of neurons in the hidden layer
epochs = 10000;

% Create networks and initialize
% Define the feedfoward net (hidden layers)
net = feedforwardnet(H, 'traingd');
% Set the input and output sizes of the net
net = configure(net, X, T);
% Initialize the weights randomly
net = init(net);

% Split 3000 samples into 1000 train, 1000 validation, and 1000 test set
% samples
net.divideParam.trainRatio = 1/3;
net.divideParam.valRatio   = 1/3;
net.divideParam.testRatio  = 1/3;

%% Training

% Specify max number of epochs that we are going to train
net.trainParam.epochs = epochs;
% Specify the max number of epochs the validation error can increase before
% doing an early stop
net.trainParam.max_fail = 200;
[net, tr] = train(net, X, T);

%% Performance of the model
hold on
plotperform(tr);
hold off

%%
% Estimate
prediction = net(X_validate);
perf = perform(net, prediction, T_validate)

prediction = net(X_train);
perf = perform(net, prediction, T_train)


%% Make a plot of the training data
% Code taken from Toledo
% Define an interpolation function for the data
f = scatteredInterpolant(X1, X2, T);

% Create a regular grid and evaluate the interpolations on it
x1 = linspace(min(X1_train), max(X1_train), 1000);
x2 = linspace(min(X2_train), max(X2_train), 1000);
[X1_mesh, X2_mesh] = meshgrid(x1, x2);
Y_mesh = f(X1_mesh, X2_mesh);

% Plot a 3D mesh of the grid with its interpolated values
figure;
hold on;
mesh(X1_mesh, X2_mesh, Y_mesh);
title('Training data');
xlabel('X1');
ylabel('X2');
zlabel('Tnew');

% Optionally add markers to get an idea of how your data is distributed
%plot3(X1_train, X2_train, T_train,'.','MarkerSize',15)


%% Create plots
figure
subplot(3,3,1);
plot(x,t,'bx',x,a11,'r',x,a21,'g'); % plot the sine function and the output of the networks
title([num2str(epochs(1)),' epochs']);
legend('target',alg1,alg2,'Location','north');
subplot(3,3,2);
postregm(a11,y); % perform a linear regression analysis and plot the result
subplot(3,3,3);
postregm(a21,y);
