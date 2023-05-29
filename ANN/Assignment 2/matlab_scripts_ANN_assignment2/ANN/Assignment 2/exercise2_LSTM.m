clear
clc
close all

%%%%%%%%%%%
%exercise2_LSTM
% A script for the solution of exercise 2, training an LSTM
% network. Inspired by the Matlab example (see openExample('nnet/TimeSeriesForecastingUsingDeepLearningExample')
%%%%%%%%%%%

%% Preamble: preprocessing data
filename = 'Data/LSTM_results.csv';
header = {'Hidden', 'Lag', 'Test MSE'};
writecell(header, filename);

% Load the datasets
train_data = load('Data/lasertrain.dat');
test_data = load('Data/laserpred.dat');

% Normalize data
mu     = mean(train_data);
sigma  = std(train_data);

% Normalize the data
dataTrainStandardized = (train_data - mu)/sigma;
dataTestStandardized = (test_data  - mu)/sigma;
 

%% Gridsearch:

% Real gridsearch
numHiddenUnits_list = [5, 10, 20, 30];
numFeatures_list = [5, 10, 20, 30];

% Or single run:
numHiddenUnits_list = 30;
numFeatures_list = 30;

nb_repetitions = 1;

for i=1:size(numHiddenUnits_list, 2)
    for j=1:size(numFeatures_list, 2)

        numHiddenUnits = numHiddenUnits_list(i);
        numFeatures = numFeatures_list(j);

        fprintf("Hidden: %d, lag: %d \n", numHiddenUnits, numFeatures);

        % Split data
        %dataTrainStandardized = dataTrainStandardized';
        [XTrain, YTrain] = getTimeSeriesTrainData(dataTrainStandardized, numFeatures);
        
        layers = [ ...
            sequenceInputLayer(numFeatures)
            lstmLayer(numHiddenUnits)
            fullyConnectedLayer(1)
            regressionLayer];
        
        % Set training options
        options = trainingOptions('adam', ...
            'MaxEpochs',500, ...
            'GradientThreshold',1, ...
            'InitialLearnRate',0.005, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',100, ...
            'LearnRateDropFactor',0.5, ...
            'Verbose', 0 ...
            ); % 'Plots','training-progress'
        
        %%% Default from the tutorial:
        %options = trainingOptions('adam', ...
        %    'MaxEpochs',250, ...
        %    'GradientThreshold',1, ...
        %    'InitialLearnRate',0.005, ...
        %    'LearnRateSchedule','piecewise', ...
        %    'LearnRateDropPeriod',125, ...
        %    'LearnRateDropFactor',0.2, ...
        %    'Verbose', 0 ...
        %    ); % 'Plots','training-progress'
        
        % Train the network a few times (for statistics)
        for a = 1:nb_repetitions

            %% Training
            disp("Training...");
            tic;
            [net, tr] = trainNetwork(XTrain,YTrain,layers,options);
            time = toc;
            disp("Time took: ");
            disp(time);
            disp("Training done!");
            
            %% Forecasting            
            net = predictAndUpdateState(net, XTrain);
            
            % Get ready for prediction
            YPred = [];
            % Get final p train data points to make the first prediction
            input_vec = dataTrainStandardized(end - numFeatures + 1 : end);
            %disp(input_vec);
        
            for b = 1:numel(dataTestStandardized)
                [net, prediction] = predictAndUpdateState(net, input_vec);
                % Save
                YPred(:, end + 1) = prediction;
                % Update input vector:
                input_vec = [input_vec(2:end); prediction];
            end
        
            % Get error on test set
            test_mse = mean(mean((YPred - dataTestStandardized).^2));
            disp("MSE:");
            disp(test_mse);

            % Save to file
            header = {numHiddenUnits, numFeatures, test_mse};
            writecell(header, filename, 'WriteMode', 'append');
        end
    end
end

disp("Done!");

% Save predicted values
writematrix(YPred, "LSTM_preds.txt");

%% Plot
figure
plot(dataTestStandardized)
hold on
plot(YPred)
hold off
legend(["Observed" "Forecast"])