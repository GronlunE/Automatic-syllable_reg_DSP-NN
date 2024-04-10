function [] = plot_history(data)
 
    % Set default font size and line width for figures
    set(groot, 'DefaultAxesFontSize', 36, ...
        'defaultAxesFontWeight','bold', ...
        'DefaultLineLineWidth', 6);

    % Extract the data from the command
    command_data = data.Command_1;
    history_data = command_data.History;

    % Plot Loss and Validation Loss
    figure("Name", "primaryLOSS");
    plot(history_data.Loss);
    hold on;
    plot(history_data.Val_Loss);
    xlabel('Epoch', 'FontAngle', 'italic');
    ylabel('Loss', 'FontAngle', 'italic');
    legend('Loss', 'Validation Loss', 'Location', 'best');

    % Plot MAE and Validation MAE
    figure("Name", "primaryMAE");
    plot(history_data.MAE);
    hold on;
    plot(history_data.Val_MAE);
    xlabel('Epoch', 'FontAngle', 'italic');
    ylabel('MAE', 'FontAngle', 'italic');
    legend('MAE', 'Validation MAE', 'Location', 'best');

    % Plot MAPE and Validation MAPE
    figure("Name","primaryMAPE");
    plot(history_data.MAPE);
    hold on;
    plot(history_data.Val_MAPE);
    xlabel('Epoch', 'FontAngle', 'italic');
    ylabel('MAPE', 'FontAngle', 'italic');
    legend('MAPE', 'Validation MAPE', 'Location', 'best');

    % Reset font size and line width to default values
    set(groot, 'DefaultAxesFontSize', 'remove', 'DefaultLineLineWidth', 'remove');
end

