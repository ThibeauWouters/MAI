echo off
clear
clc
close all


%%%%%%%%%%%%%%%%%
% exercise2_SAE %
%%%%%%%%%%%%%%%%%

%% Load and preprocess datasets

% Load train and test datasets
load('Files/digittrain_dataset.mat');
load('Files/digittest_dataset.mat');

create_plots = false;
show_window = false;
% Fix random seed, or not?
%rng('default')

% To save data
filename = 'Files/SAE_results.csv';

% To initialize the file again (now, we keep on appending over several
% runs)
%header = {'Hidden', 'Layers', 'Acc before', 'Acc after'};
%writecell(header, filename);

% Get the number of pixels in each image
imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;

% Turn the training images into vectors and put them in a matrix
xTrain = zeros(inputSize, numel(xTrainImages));

for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
end

% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize, numel(xTestImages));

for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end


%% Train the autoencoders and final softmax layer:

% Specify the stacked autoencoder architecture through the hidden layer
% sizes:
%h =  [200, 100, 50, 25];

h = [100, 50];
num_iterations = 200;
net = feedforwardnet(h, "traincgf");
% Save the number of epochs to train
net.trainParam.epochs = num_iterations;
% Split into validation and training set for early stopping
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio   = 0.2;
net.divideParam.testRatio  = 0;
% Save the number of validation checks before early stopping 
net.trainParam.max_fail = 10;

% Set hidden layer activation functions to tansig
num_hidden_layers = numel(net.layers) - 1;
for i = 1:num_hidden_layers
    net.layers{i}.transferFcn = 'logsig';
end

% Output layer also logsig:
net.layers{end}.transferFcn = 'softmax';

%% Train
net.trainParam.showWindow = 0;
[net, tr] = train(net, xTrain, tTrain);

y = net(xTest);

[~, targets] = max(tTest);
[~, predictions] = max(y);
accuracy = sum(predictions == targets)/length(predictions);
disp("Accuracy: " + accuracy);