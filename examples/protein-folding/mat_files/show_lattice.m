function show_lattice_new(ising_string, latdim, hp_sequence, keys)
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
    axes1 = axes;
    imagesc(image);
    colormap(gca, lattice_colors);
    set(gca, 'XTick', 1:latdim(2), 'XTickLabel', 0:latdim(2)-1, ...
             'YTick', 1:latdim(1), 'YTickLabel', 0:latdim(1)-1);
    axis equal;
    hold on;

    seqlen = length(hp_sequence);
    fpos = [];
    xpos = [];
    ypos = [];
    posc = [];
    xstart = [];
    ystart = [];
    cstart = [];
    text_dict = containers.Map;
    
    for i = 1:length(ising_string)
        if ising_string(i) ~= 1
            continue;
        end
        s = keys{i, 1};
        f = keys{i, 2} + 1; % +1 to convert python idx into matlab
        s1 = s(1) + 1; % +1 to convert python idx into matlab
        s2 = s(2) + 1; % +1 to convert python idx into matlab

        fpos = [fpos, f];
        xpos = [xpos, s1];
        ypos = [ypos, s2];
        posc = [posc; hp_colors(hp_sequence(f) + 1, :)];
        if f == 1
            xstart = [xstart, s1];
            ystart = [ystart, s2];
            cstart = [cstart, hp_sequence(f)];
        end

        key = mat2str([s1, s2]);
        if isKey(text_dict, key)
            text_dict(key) = [text_dict(key), f];
        else
            text_dict(key) = [f];
        end
    end

    dict_keys = text_dict.keys;
    for i = 1:length(dict_keys)
        key = dict_keys{i};
        k = str2num(key); %#ok<ST2NM>
        v = text_dict(key);
        t = strjoin(arrayfun(@num2str, v, 'UniformOutput', false), ',');
        text(k(1)-0.4, k(2)-0.3, t, 'Color', 'k', 'FontSize', 8, 'HorizontalAlignment', 'left', 'Parent', axes1);
    end
    
    [~, idx] = sort(fpos);
    xpos = xpos(idx);
    ypos = ypos(idx);
    posc = posc(idx, :);
    plot(axes1, xpos, ypos, 'k-', 'LineWidth', 0.5);
    scatter(axes1, xpos, ypos, 100, posc, 'filled', 'MarkerFaceAlpha', 0.5);
    scatter(axes1, xstart, ystart, 25, 'k', 'filled', 'Marker', 'v');
    
    hold off;
end

function p = parity(pos)
    % define parity function here if needed, otherwise assume it exists
    p = mod(mod(pos(1),2) + mod(pos(2),2), 2);
end
