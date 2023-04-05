clear
clc
close all

%%%%%%%%%%%
%exercise2_sine
% A script for the solution of exercise 2, training a recurrent neural
% network
%%%%%%%%%%%

train_data = load('lasertrain.dat');
test_data = load('laserpred.dat');

% Compute mean and std
mu     = mean(train_data);
sigma  = std(train_data);

% Normalize data
train_data = (train_data - mu)/sigma;
test_data  = (test_data  - mu)/sigma;

% Create a new CSV file for saving the data
filename = 'RNN_results.csv';
header = {'Hidden', 'p', 'Train MSE', 'Test MSE'};
writecell(header, filename);

%%%  Define the hyperparams

% Lag parameter
p_list = [20];
% Size hidden layer
hidden_layer_size_list = [50];
% Max number of training epochs
num_iterations = 500;
% How many epochs to wait before early stopping
max_fail = 100;
% How often to repeat each setup
nb_repetitions = 3;

%%% Train the networks

% Vary over lag parameter
for a = 1:length(p_list)
    p = p_list(a);
    % Get training data for the RNN
    [train_data_rnn, train_target] = getTimeSeriesTrainData(train_data, p);
    % Vary over the size of the hidden layer
    for b = 1:length(hidden_layer_size_list)
        hidden_layer_size = hidden_layer_size_list(b);
        fprintf('Training for hidden size %d and p %d \n', hidden_layer_size, p);
        for i = 1:nb_repetitions
            fprintf('--- Training %d out of %d\n', i, nb_repetitions);
            net = feedforwardnet(hidden_layer_size , "trainlm");
            % Configure to the data
            net = configure(net, train_data_rnn, train_target);
            % Split into validation and training set for early stopping
            net.divideFcn = 'divideint';
            net.divideParam.trainRatio = 0.7;
            net.divideParam.valRatio   = 0.3;
            net.divideParam.testRatio  = 0;
            % Save the number of validation checks before early stopping 
            net.trainParam.max_fail = 10;
            % Randomly initialize the weights
            net = init(net);
            % Don't show window during training (annoying)
            net.trainParam.showWindow = 0;
            % Save the number of epochs to train
            net.trainParam.epochs = num_iterations;
            % Set hidden layer activation functions to tansig
            num_hidden_layers = numel(net.layers) - 1;
            for k = 1:num_hidden_layers
                net.layers{k}.transferFcn = 'tansig';
            end
            
            % Train the network
            [net, tr] = train(net, train_data_rnn, train_target);
            
            %%% Plot training procedure for insights
            %plotperform(tr);
            
            %%% Predict on the test set
            
            % Get the initial input, i.e. final values of the train set
            len_train_data = size(train_data_rnn, 2);
            input_vec = train_data_rnn(:, len_train_data);
            
            % Get the number of predictions to be made
            num_predictions = size(test_data, 1);
            
            % Initialize the output vector with zeros
            output_vec = zeros(size(test_data));
            
            % Loop over the remaining timesteps of the test set
            for l = 2:num_predictions
                % Predict the next timestep using the current input vector
                predicted_val = net(input_vec);
            
                % Store the predicted value in the output vector
                output_vec(l) = predicted_val;
            
                % Update the input vector by shifting in the predicted value
                input_vec = [input_vec(2:end);predicted_val];
            end
            
            % Compute MSE on the test set
            mse = mean((test_data - output_vec).^2);
            % Write down observations to CSV file
            results = {hidden_layer_size, p, tr.best_vperf, mse};
            writecell(results, filename, 'WriteMode', 'append');
        end
    end
end

% Plot the predicted output against the actual test set
figure;
plot(test_data, 'b');
hold on;
plot(output_vec, 'r');
legend('Actual', 'Predicted');
disp("Done!");