function show_lattice(ising_string, latdim, hp_sequence, keys)
    image = zeros(latdim);
    for i = 1:latdim(1)
        for j = 1:latdim(2)
            image(i, j) = parity([i-1, j-1]); % Assuming parity is a function available
        end
    end

    lattice_colors = [234, 234, 234; 254, 254, 254] / 255;
    hp_colors = [17, 240, 51; 240, 51, 17] / 255;
    
    % Plot the lattice
    figure;
    imagesc(image);
    colormap(gca, lattice_colors);
    set(gca, 'XTick', 1:latdim(2), 'XTickLabel', 0:latdim(2)-1, ...
             'YTick', 1:latdim(1), 'YTickLabel', 0:latdim(1)-1);
    axis equal;
    hold on;

    seqlen = length(hp_sequence);
    xpos = zeros(1, seqlen);
    ypos = zeros(1, seqlen);
    posc = zeros(seqlen, 3);
    xstart = [];
    ystart = [];
    cstart = [];
    
    for i = 1:length(ising_string)
        if ising_string(i) == -1
            continue;
        end
        s = keys{i, 1};
        f = keys{i, 2} + 1; % +1 to convert python idx into matlab
        s1 = s(1) + 1; % +1 to convert python idx into matlab
        s2 = s(2) + 1; % +1 to convert python idx into matlab
        xpos(f) = s1;
        ypos(f) = s2;
        posc(f, :) = hp_colors(hp_sequence(f) + 1, :); % get rgb triplet for the bead colour
        if f == 1
            xstart = [xstart, s1];
            ystart = [ystart, s2];
            cstart = [cstart, hp_sequence(f)];
        end
    end
    
    scatter(xpos, ypos, 100, posc, 'filled', 'MarkerEdgeColor', 'k');
    plot(xpos, ypos, 'k-');
    scatter(xstart, ystart, 25, 'k', 'filled', 'Marker', 'v');
    
    hold off;
end

function p = parity(pos)
    % define parity function here if needed, otherwise assume it exists
    p = mod(mod(pos(1),2) + mod(pos(2),2), 2);
end