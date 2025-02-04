function [magnetization, energy] = pbit_n(iteration, coupling_matrix, external_field, magnetization, inverse_temperature, energy)
    % PBIT_N Performs one iteration of the p-bit algorithm
    %   [magnetization, energy] = pbit_n(iteration, coupling_matrix, external_field, magnetization, inverse_temperature, energy)
    %
    %   Inputs:
    %   - iteration: Current iteration number
    %   - coupling_matrix: Matrix of coupling strengths between p-bits
    %   - external_field: Vector of external field strengths for each p-bit
    %   - magnetization: Matrix of magnetization values for all iterations
    %   - inverse_temperature: Inverse temperature parameter (beta)
    %   - energy: Vector of energy values for all iterations
    %
    %   Outputs:
    %   - magnetization: Updated magnetization matrix
    %   - energy: Updated energy vector
    
    %% Initialize current iteration using values from previous iteration
    if iteration > 1
        magnetization(iteration, :) = magnetization(iteration-1, :);
    end

    %% Update magnetization for each p-bit
    num_pbits = size(coupling_matrix, 1);
    for i = 1:num_pbits
        % Calculate effective field (Ii) for p-bit i
        effective_field = 0;
        for j = 1:num_pbits
            effective_field = effective_field + coupling_matrix(i,j) * magnetization(iteration, j);
        end
        effective_field = effective_field + external_field(i);
        
        % Generate random number for probabilistic update
        random_value = 2 * rand(1) - 1;  % Random number between -1 and 1
        
        % Update magnetization of p-bit i
        magnetization(iteration, i) = sign(tanh(inverse_temperature * effective_field) - random_value);
    end

    %% Calculate system energy
    total_energy = 0;

    % Contribution from coupling between p-bits
    for i = 1:num_pbits
        for j = 1:num_pbits
            total_energy = total_energy + 0.5 * coupling_matrix(i,j) * magnetization(iteration, i) * magnetization(iteration, j);
        end
    end

    % Contribution from external field
    for i = 1:length(external_field)
        total_energy = total_energy + external_field(i) * magnetization(iteration, i);
    end

    % Store negative of total energy (Hamiltonian convention)
    energy(iteration) = -total_energy;
end

