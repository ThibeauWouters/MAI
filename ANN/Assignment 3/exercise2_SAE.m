clear
clc
close all

%%%%%%%%%%%
% exercise2_SAE
% A script for the solution of exercise 2, stacked autoencoders
%%%%%%%%%%%

%% Load and preprocess datasets

% Load train and test datasets
load('digittrain_dataset.mat');
load('digittest_dataset.mat');

% Get the number of pixels in each image
imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;

% Turn the training images into vectors and put them in a matrix
xTrain = zeros(inputSize, numel(xTrainImages));

echo off;
for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
end
echo on;

% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize, numel(xTestImages));

echo off
for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end
echo on

% Fix random seed, or not?
rng('default')

% Boolean variable: not finetuned yet
finetuned = 0;

%% Train the autoencoders and final softmax layer:

% Specify the stacked autoencoder architecture through the hidden layer
% sizes:
hiddenSizeList = [100, 50];
autoencList = {};
% First features are just training data
features = xTrainImages;
% Hyperparams:
MaxEpochsList = [200, 200];
L2WeightRegularizationList = [0.004, 0.002];
SparsityRegularizationList = [4, 4];
SparsityProportionList = [0.15, 0.11];

for i=1:length(hiddenSizeList)
    % Define the next autoencoder and train it
    autoenc = trainAutoencoder(features, hiddenSizeList(i), ...
    'MaxEpochs', MaxEpochsList(i), ...
    'L2WeightRegularization', L2WeightRegularizationList(i), ...
    'SparsityRegularization', SparsityRegularizationList(i), ...
    'SparsityProportion', SparsityProportionList(i), ...
    'ScaleData', false);

    % Save the autoencoder
    autoencList{end+1} = autoenc;

    % Get the features for the next layer
    features = encode(autoenc, features);

end

% Final softmax layer
softmaxEpochs = 300;

softnet = trainSoftmaxLayer(features, tTrain,'MaxEpochs', softmaxEpochs);

% Add it to "autoencoderlist", not an autoencoder, but saved for stacking
autoencList{end+1} = softnet;

%%% To inspect the abstract features:
%figure;
%plotWeights(autoencList{1});

%% Get deep network

deep = stack(autoencList{1}, autoencList{2});

% If there are many networks, keep on stacking
if length(autoencList) > 2
    for i=3:length(autoencList)
        deep = stack(deep, autoencList{i});
    end
end

% View the final, stacked network
%view(deep);

%% Predictions on the test set

y = deep(xTest);
figure;
plotconfusion(tTest, y);

% Get the accuracy: turn vectors into arrays of integers from 1 to 10
[~, targets] = max(tTest);
[~, predictions] = max(y);
accuracy = sum(predictions == targets)/length(predictions);
disp("Accuracy before finetuning: " + accuracy);

%% Finetuning

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'InitialLearnRate', 0.001);

% Perform fine tuning, specify training options
% TODO early stopping? 

deep = train(deep, xTrain, tTrain);
finetuned = 1;

%% Get the final predictions
y = deep(xTest);
figure;
plotconfusion(tTest,y);

% Get the accuracy: turn vectors into arrays of integers from 1 to 10
[~, targets] = max(tTest);
[~, predictions] = max(y);
accuracy = sum(predictions == targets)/length(predictions);
disp("Accuracy after finetuning: " + accuracy);

%%% To save data
%filename = 'RNN_results.csv';
%header = {'Hidden', 'p', 'Train MSE', 'Val MSE', 'Test MSE'};
%writecell(header, filename);
