function eval_langdep(data)
% Get all command field names and language combinations
command_fields = fieldnames(data);


% Preallocate language_combinations cell array
language_combinations = cell(1, 6);
history_data = struct('Loss', {}, 'MAE', {}, 'MAPE', {}, 'Val_Loss', {}, 'Val_MAE', {}, 'Val_MAPE', {});

% Loop over all Commands
for n = 2:numel(data.Command_2)
    % Extract language combination from Call field of Command
    language = data.(['Command_' num2str(n)]).Call{2};
    
    % Add the language combination to language_combinations
    language_combinations{n-1} = language;
    
    % Extract history data from History field of Command
    history = data.(['Command_' num2str(n)]).History;
    
    % Add history data to history_data
    history_data{n-1, 1} = history.Loss;
    history_data{n-1, 2} = history.MAE;
    history_data{n-1, 3} = history.MAPE;
    history_data{n-1, 4} = history.Val_Loss;
    history_data{n-1, 5} = history.Val_MAE;
    history_data{n-1, 6} = history.Val_MAPE;
end


% Plot history for each command field
figure('Name', 'History Plots');
for i = 1:length(command_fields)
    % Get command data
    command_data = data.(command_fields{i});

    % Plot history for each metric
    for j = 1:length(command_data.History)
        subplot(length(command_fields), length(command_data.History), (i-1)*length(command_data.History)+j);
        hold on;
        
        % Plot train and validation history for each language combination
        for k = 1:length(language_combinations)
            language_data = get_language_data(command_data.History(j), language_combinations{k});
            plot(1:length(language_data.train), language_data.train, 'LineWidth', 2, 'DisplayName', language_combinations{k});
            plot(1:length(language_data.val), language_data.val, 'LineWidth', 2, 'DisplayName', [language_combinations{k} ' (Val)']);
        end
        
        % Set plot title and axis labels
        title([command_fields{i} ' ' command_data.History(j).Name], 'Interpreter', 'none');
        xlabel('Epoch');
        ylabel(command_data.History(j).Name);
        legend('Location', 'Best');
    end
end

% Plot predictions for each command field and language combination
figure('Name', 'Prediction Plots');
for i = 1:length(command_fields)
    % Get command data
    command_data = data.(command_fields{i});

    % Plot predictions for each language combination
    for j = 1:length(language_combinations)
        language_data = get_language_data(command_data.Predictions, language_combinations{j});

        subplot(length(command_fields), length(language_combinations), (i-1)*length(language_combinations)+j);
        bar([language_data.english.MAE language_data.estonian.MAE]);
        xticklabels({'English', 'Estonian'});
        ylabel('MAE');
        title({[command_fields{i} ' ' language_combinations{j}]; ['MAE: ' num2str(language_data.english.MAE) ', ' num2str(language_data.estonian.MAE)]}, 'Interpreter', 'none');
    end
end

end

function language_data = get_language_data(data, language_combination)
% Get language data for a specific language combination
language_data = struct();
for lang = split(language_combination, ', ')
    lang = lang{:};
    language_data.(lang) = data.(lang);
end
end
