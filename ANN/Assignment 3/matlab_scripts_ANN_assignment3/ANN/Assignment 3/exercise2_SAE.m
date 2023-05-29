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
%hiddenSizeList = [100, 50];
hiddenSizes =  [200];
layers = [1, 2, 3];

% Other hyperparams:
MaxEpochsList = [200, 200, 200];
L2WeightRegularizationList = [0.004, 0.002, 0.001];
SparsityRegularizationList = [4, 4, 4];
SparsityProportionList = [0.15, 0.11, 0.09];

time_taken = 0;
nb_repetitions = 3;
for a=1:nb_repetitions
    fprintf("Time taken: %f\n", time_taken)
    tic;
    fprintf("=========== Rep %d of %d \n", a, nb_repetitions)
    for i = 1:length(layers)
        for j=1:length(hiddenSizes)
            % Empty some vars
            autoencList = {};
            features = xTrainImages;
            
            % Determine size
            N = layers(i);
            h = hiddenSizes(j);
            %hiddenSizeList = repelem([h], N);
            hiddenSizeList = zeros(1, N);
            for c=1:N
                hiddenSizeList(c) = h / (2.^ (c-1));
            end

            % Define the next autoencoder and train it
            disp("Hidden size list:");
            disp(hiddenSizeList);
            fprintf("Training autoencoder %d of %d \n", i, numel(hiddenSizeList));
            

            autoenc = trainAutoencoder(features, hiddenSizeList(i), ...
            'MaxEpochs', MaxEpochsList(i), ...
            'L2WeightRegularization', L2WeightRegularizationList(i), ...
            'SparsityRegularization', SparsityRegularizationList(i), ...
            'SparsityProportion', SparsityProportionList(i), ...
            'ScaleData', false, ...
            'ShowProgressWindow', show_window);
        
            % Save the autoencoder
            autoencList{end+1} = autoenc;
        
            % Get the features for the next layer
            features = encode(autoenc, features);

            
            disp("Autoencoders trained!");
          
            % Final softmax layer
            softmaxEpochs = 300;
            
            disp("Training softmax");
            softnet = trainSoftmaxLayer(features, tTrain,'MaxEpochs', softmaxEpochs, 'ShowProgressWindow', show_window);
            disp("Softmax done!");
            
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
            if create_plots
                figure;
                plotconfusion(tTest, y);
            end
            
            % Get the accuracy: turn vectors into arrays of integers from 1 to 10
            [~, targets] = max(tTest);
            [~, predictions] = max(y);
            accuracy = sum(predictions == targets)/length(predictions);
            disp("Accuracy before finetuning: " + accuracy);
            
            accuracy_before = accuracy;
            
            %% Finetuning
            
            options = trainingOptions('adam', ...
                'MaxEpochs', 100, ...
                'InitialLearnRate', 0.001 ...
                );
            
            % Perform fine tuning, specify training options
            % TODO early stopping? 
            
            disp("Finetuning...")
            deep.trainParam.showWindow = show_window;
            deep = train(deep, xTrain, tTrain);
            disp("Finetuning done!")
            
            %% Get the final predictions
            y = deep(xTest);
            if create_plots
                figure;
                plotconfusion(tTest,y);
            end
            
            % Get the accuracy: turn vectors into arrays of integers from 1 to 10
            [~, targets] = max(tTest);
            [~, predictions] = max(y);
            accuracy = sum(predictions == targets)/length(predictions);
            disp("Accuracy after finetuning: " + accuracy);
            
            accuracy_after = accuracy;
            
            %%% To save data
            header = {h, N, accuracy_before, accuracy_after};
            writecell(header, filename, 'WriteMode', 'append');
            time_taken=toc;
        end
    end
end