clear
clc
close all

%%%%%%%%%%%
%exercise2_LSTM
% A script for the solution of exercise 2, training an LSTM
% network. Inspired by the Matlab example (see openExample('nnet/TimeSeriesForecastingUsingDeepLearningExample')
%%%%%%%%%%%

% file for saving results
filename = 'LSTM_results.csv';
header = {'Hidden', 'Train MSE', 'Test MSE'};
writecell(header, filename);

% Load the datasets
train_data = load('Data/lasertrain.dat');

% Note: test_data is the real test data, while dataTest is the validation
% set!
test_data = load('Data/laserpred.dat');

% We transpose the data, such that it has the same shape as the example
train_data = train_data';
test_data = test_data';

% Train on the first 90%, validate on final 10%
%numTimeStepsTrain = floor(1.0*numel(train_data));

%dataTrain = train_data(1:numTimeStepsTrain+1);
% dataTest = train_data(numTimeStepsTrain+1:end);

% Normalize data
mu     = mean(train_data);
sigma  = std(train_data);

% Normalize the data
dataTrainStandardized = (train_data - mu)/sigma;
dataTestStandardized = (test_data  - mu)/sigma;

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

% Define the LSTM
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 500;
nb_epochs = 100;
nb_repetitions = 1;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs', nb_epochs, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',100, ...
    'LearnRateDropFactor',0.5, ...
    'Verbose',0, ...
    'Plots','training-progress'); % 

% Train the network a few times for statistics
for a = 1:nb_repetitions
    disp("Training...");
    [net, tr] = trainNetwork(XTrain,YTrain,layers,options);
    disp("Training done!");
    
    %%% Forecasting
    
    XTest = dataTestStandardized(1:end-1);
    YTest = dataTestStandardized(2:end);
    
    net = predictAndUpdateState(net, XTrain);
    [net,YPred] = predictAndUpdateState(net,YTrain(end));
    
    numTimeStepsTest = numel(XTest);
    for i = 2:numTimeStepsTest
        [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
    end
    
    mse = mean((YPred-YTest).^2);
    disp(mse);
    results = {numHiddenUnits, tr.TrainingRMSE(end).^2, mse};
                writecell(results, filename, 'WriteMode', 'append');
end
% plot

figure
%subplot(2,1,1)
plot(YTest,'.-')
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

