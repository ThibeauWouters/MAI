clear
clc
close all

%%%%%%%%%%%
%exercise2_LSTM
% A script for the solution of exercise 2, training an LSTM
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
%filename = 'LSTM_results.csv';
%header = {'Hidden', 'p', 'Train MSE', 'Test MSE'};
%writecell(header, filename);