clear
clc
close all

%%%%%%%%%%%
%exercise2_RNN
% A script for the solution of exercise 2, training a recurrent neural
% network
%%%%%%%%%%%

train_data = load('Data/lasertrain.dat');
test_data = load('Data/laserpred.dat');

% Compute mean and std
mu     = mean(train_data);
sigma  = std(train_data);

% Normalize data, use same transformation on the test data as well
train_data = (train_data - mu)/sigma;
test_data  = (test_data  - mu)/sigma;

% Create a new CSV file for saving the data
filename = 'RNN_results.csv';
header = {'Hidden', 'p', 'Train MSE', 'Val MSE', 'My val MSE', 'Test MSE'};
writecell(header, filename);

%%%  Define the hyperparams

% Lag parameter
p_list = [5];
%p_list = [40, 50];
% Size hidden layer
hidden_layer_size_list = [20];
%hidden_layer_size_list = [40, 50];
% Max number of training epochs
num_iterations = 1000;
% How many epochs to wait before early stopping
max_fail = ceil(0.1*num_iterations);
% How often to repeat each setup
nb_repetitions = 1;

% For tuning, get my own validation set
my_validation_set_start = 544;
my_validation_set = train_data(my_validation_set_start:my_validation_set_start + 99);

%% Train the networks



% Vary over lag parameter
for a = 1:length(p_list)
    p = p_list(a);
    % Get training data for the RNN
    [train_data_rnn, train_target] = getTimeSeriesTrainData(train_data, p);
    train_ratio = 0.7; 
    num_data = size(train_data_rnn, 2);
    all_indices = 1:num_data;
    train_max_index = floor(num_data * train_ratio);
    train_indices   = all_indices(1:train_max_index);
    val_indices     = all_indices(train_max_index+1:end);
    % Vary over the size of the hidden layer
    for b = 1:length(hidden_layer_size_list)
        hidden_layer_size = hidden_layer_size_list(b);
        fprintf('Training for hidden size %d and p %d \n', hidden_layer_size, p);
        for i = 1:nb_repetitions
            fprintf('--- Training %d out of %d\n', i, nb_repetitions);
            net = feedforwardnet(hidden_layer_size , "trainbr");
            % Configure to the data
            net = configure(net, train_data_rnn, train_target);
            % Split into validation and training set for early stopping
            net.divideFcn = 'divideind';
            net.divideParam.trainInd = train_indices;
            net.divideParam.valInd   = val_indices;
            net.divideParam.testInd  = [];   % No test set
            % Save the number of validation checks before early stopping 
            net.trainParam.max_fail = max_fail;
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

            % Get the initial input, i.e. final values of the train set
            len_train_data = size(train_data_rnn, 2);

            %%% Predict on my validation set 

            input_vec = train_data(my_validation_set_start-p+1:my_validation_set_start);
            
            % Get the number of predictions to be made
            num_predictions = size(my_validation_set, 1);
            
            % Initialize the output vector with zeros
            output_vec = zeros(size(my_validation_set));
            
            % Loop over the remaining timesteps of the test set
            for l = 1:num_predictions
                % Predict the next timestep using the current input vector
                predicted_val = net(input_vec);
                %disp("Prediction");
                %disp(predicted_val);
            
                % Store the predicted value in the final output vector
                output_vec(l) = predicted_val;
            
                % Shift input vector and append the predicted value
                input_vec = [input_vec(2:end);predicted_val];
                %disp("New input");
                %disp(input_vec);
            end
            
            % Compute MSE on the test set
            my_val_mse = mean((my_validation_set - output_vec).^2);
            

            %%% Predict on the real test set

            input_vec = train_data(end-p+1:end);
            
            % Get the number of predictions to be made
            num_predictions = size(test_data, 1);
            
            % Initialize the output vector with zeros
            output_vec = zeros(size(test_data));
            
            % Loop over the remaining timesteps of the test set
            for l = 1:num_predictions
                % Predict the next timestep using the current input vector
                predicted_val = net(input_vec);
                %disp("Prediction");
                %disp(predicted_val);
            
                % Store the predicted value in the final output vector
                output_vec(l) = predicted_val;
            
                % Shift input vector and append the predicted value
                input_vec = [input_vec(2:end);predicted_val];
                %disp("New input");
                %disp(input_vec);
            end
            
            % Compute MSE on the test set
            mse = mean((test_data - output_vec).^2);
            % Write down observations to CSV file
            results = {hidden_layer_size, p, tr.best_perf, tr.best_vperf, my_val_mse, mse};
            writecell(results, filename, 'WriteMode', 'append');
        end
    end
end

% Plot the predicted output against the actual test set
%figure;
%plot(test_data, 'b');
%hold on;
%plot(output_vec, 'r');
%legend('Actual', 'Predicted');
disp("Done!");

% Write the output
%writematrix(output_vec, "RNN_output.txt");