function Genfig_Fact_one(mi, E)
    % Convert columns to logical indices
    m1_in1 = int8(mi(:,1) == 1);
    m2_in2 = int8(mi(:,2) == 1);
    m3_out1 = int8(mi(:,3) == 1);
    m4_out2 = int8(mi(:,4) == 1);
    
    % Calculate values based on the binary inputs and outputs
    val1 = 2*(2*(2*1 + m1_in1) + m2_in2) + 1; 
    val2 = 2*(2*(2*1 + m3_out1) + m4_out2) + 1; 
    val = [val1, val2];

    figure, subplot(211), plot(val)
    
    % Sort the pairs so that symmetric pairs are treated as identical
    sorted_val = sort(val, 2);
    
    % Define all possible outcomes (adjust the range as necessary)
    range = 1:15;  % All numbers between 1 and 15
    [X, Y] = meshgrid(range, range);
    possible_outcomes = sort([X(:), Y(:)], 2);  % Create all combinations of pairs and sort them
    
    % Convert sorted observed outcomes to string for easy comparison
    observed_str = arrayfun(@(x) sprintf('{%d,%d}', sorted_val(x, 1), sorted_val(x, 2)), 1:size(sorted_val, 1), 'UniformOutput', false);
    
    % Convert sorted possible outcomes to string
    possible_str = arrayfun(@(x) sprintf('{%d,%d}', possible_outcomes(x, 1), possible_outcomes(x, 2)), 1:size(possible_outcomes, 1), 'UniformOutput', false);
    
    % Count the frequency of each possible outcome in the observed data
    freq_map = containers.Map;  % Use a map to accumulate frequencies for unique pairs
    for i = 1:length(observed_str)
        if isKey(freq_map, observed_str{i})
            freq_map(observed_str{i}) = freq_map(observed_str{i}) + 1;
        else
            freq_map(observed_str{i}) = 1;
        end
    end
    
    % Extract keys (labels) and values (frequencies)
    labels = keys(freq_map);
    freq = cell2mat(values(freq_map));
    
    % Convert labels to categorical for plotting
    labels = categorical(labels);
    
    % Plot the histogram
    subplot(212), 
    bar(labels, freq, 'FaceColor', [0, 0.5, 0.8]);
    xlabel('Possible Outcomes');
    ylabel('Frequency');
    title('Histogram of Combined Symmetric Pairs');
    xtickangle(90);  % Rotate x-axis labels for better readability
end
