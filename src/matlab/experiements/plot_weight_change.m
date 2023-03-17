function plot_weight_change(weights, biases)
%% Function description

% Plot change in weights and biases across layers using line plots, grouped bar charts, and box plots
%
% Inputs:
%   - weights: a cell array of weight tensors for each layer in a neural network
%   - biases: a cell array of bias tensors for each layer in a neural network

%% Setup

% Compute the number of layers in the network
num_layers = length(weights);

% Extract the weight and bias values for each layer
weight_values = zeros(num_layers, 1);
bias_values = zeros(num_layers, 1);
for i = 1:num_layers
    weight_values(i) = mean(mean(abs(weights{i})));
    bias_values(i) = mean(mean(abs(biases{i})));
end

%% Plot weight values using line plot

subplot(1, 3, 1)
plot(weight_values, '-o')
xlabel('Layer')
ylabel('Weight value')
title('Change in weight values across layers (line plot)')

%% Plot bias values using line plot

subplot(1, 3, 2)
plot(bias_values, '-o')
xlabel('Layer')
ylabel('Bias value')
title('Change in bias values across layers (line plot)')

%% Plot weight and bias values using grouped bar chart

subplot(1, 3, 3)
bar_values = [weight_values, bias_values];
bar(bar_values)
xlabel('Layer')
ylabel('Value')
legend('Weights', 'Biases')
title('Change in weight and bias values across layers (grouped bar chart)')

%% Plot weight and bias values using box plot

figure
boxplot(bar_values)
xlabel('Value')
ylabel('Type')
set(gca, 'XTick', [1 2], 'XTickLabel', {'Weights', 'Biases'})
title('Distribution of weight and bias values across layers (box plot)')

end
