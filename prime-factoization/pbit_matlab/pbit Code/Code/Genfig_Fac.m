function Genfig_Fac(mi, E)
    % Convert columns to logical indices
    m1_in1 = int8(mi(:,1) == 1);
    m2_in2 = int8(mi(:,2) == 1);
    m3_out1 = int8(mi(:,3) == 1);
    m4_out2 = int8(mi(:,4) == 1);

    % Calculate values based on the binary inputs and outputs
    val = (2*m1_in1 + 1); 
    % Count occurrences of each value
    aa = histcounts(val, 0:4);
    % Define labels correctly (removed the duplicate '1000')
    lbl = categorical({'00', '01', '10', '11'});
   % Plot the bar graph for all states
    figure, subplot(211), bar(lbl, aa, 0.4, 'FaceColor', [0 .5 .5]);
    title("Probabilistic AND Gate (PAG)");

    
     % Calculate values based on the binary inputs and outputs
    val = 2*(2*m2_in2 + m3_out1)+1; 
    % Count occurrences of each value
    aa = histcounts(val, 0:8);
    % Define labels correctly (removed the duplicate '1000')
    lbl = categorical({'000', '001', '010', '011','100','101','110','111'});
    % Plot the bar graph for all states
    subplot(212), bar(lbl, aa, 0.4, 'FaceColor', [0 .5 .5]);
    title("Probabilistic AND Gate (PAG)");

    % Filter values for the minimum energy state
    m1_in1(E ~= min(E)) = [];
    m2_in2(E ~= min(E)) = [];
    m3_out1(E ~= min(E)) = [];
    m4_out2(E ~= min(E)) = [];

    % Recalculate values for the filtered inputs/outputs
    val = 2*(2*(2*m1_in1 + m2_in2) + m3_out1) + m4_out2;
    
    % Count occurrences of each value
    aa = histcounts(val, 0:16);
    lbl = categorical({'0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111', ...
                       '1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111'});

    % Plot the bar graph for minimum energy states only
    figure, subplot(211), bar(lbl, aa, 0.4, 'FaceColor', [0 .5 .5]);
 %   title("Minimum Energy State Only: PAG");
end



