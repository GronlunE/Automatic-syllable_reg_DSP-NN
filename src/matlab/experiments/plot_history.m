function plot_history(history_dict)
%% Extract the history data

loss = history_dict.Loss;
mae = history_dict.MAE;
mape = history_dict.MAPE;
val_loss = history_dict.Val_Loss;
val_mae = history_dict.Val_MAE;
val_mape = history_dict.Val_MAPE;

%% Plot the training and validation loss

figure;
subplot(3,1,1)
plot(loss);
hold on;
plot(val_loss);
legend('Training Loss', 'Validation Loss');
xlabel('Epoch');
ylabel('Loss');
title('Training and Validation Loss');

%% Plot the training and validation MAE

subplot(3,1,2)
plot(mae);
hold on;
plot(val_mae);
legend('Training MAE', 'Validation MAE');
xlabel('Epoch');
ylabel('MAE');
title('Training and Validation MAE');

%% Plot the training and validation MAPE

subplot(3,1,3)
plot(mape);
hold on;
plot(val_mape);
legend('Training MAPE', 'Validation MAPE');
xlabel('Epoch');
ylabel('MAPE');
title('Training and Validation MAPE');
end