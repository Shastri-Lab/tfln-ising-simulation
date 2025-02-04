clc
clear all
close all
% Example usage
size = 16;
Max_number = 16;
%nset = randi(Max_number, 1, size)
function nset = generate_even_sum_set(size, max_value)
    % Generate a random number set with an even sum
    nset = randi([1, max_value], 1, size);
    if mod(sum(nset), 2) ~= 0
        % If the sum is odd, adjust the last element to make the sum even
        if nset(end) == max_value
            nset(end) = nset(end) - 1;
        else
            nset(end) = nset(end) + 1;
        end
    end
end
nset = generate_even_sum_set(size, Max_number);
% Generate the J matrix for the number partitioning problem.
N = length(nset);
J = zeros(N, N);
nset1 = nset;
nset = nset/max(nset);
% Generate the J matrix for the number partitioning problem.
N = length(nset);
J = zeros(N, N);
for i = 1:N
    for j = i+1:N
        J(i, j) =  nset(i) * nset(j);
        J(j, i) = J(i, j);  % J is symmetric
    end
end
config = randi([0, 1], 1, size) * 2 - 1;  % Spin configuration
difference = abs(sum(nset.*config))
cost = difference * difference
energy = .5 * config * J * config'