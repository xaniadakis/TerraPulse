function [Bx, By, nr] = read_dat_file(fn)
    fid = fopen(fn, 'rb');
    if fid == -1
        error('read_dat_file: File opening failed');
    end

    fseek(fid, 64, 'bof');
    fseek(fid, 0, 'eof');
    file_size = ftell(fid) - 64;
    fseek(fid, 64, 'bof');

    data = fread(fid, file_size, 'uint8');
    fclose(fid);

    ld = file_size;
    Bx_size = ld / 4;
    By_size = ld / 4;

    Bx = zeros(Bx_size, 1, 'int32');
    By = zeros(By_size, 1, 'int32');

    a = 65536;
    nr = 0;
    i = 1;

    % First loop for 89 iterations
    for j = 1:89
        Bx(nr + 1) = a * bitshift(bitand(data(i), 12), -2) + data(i+1) * 256 + data(i+2);
        By(nr + 1) = a * bitand(data(i), 3) + data(i+3) * 256 + data(i+4);
        nr = nr + 1;
        i = i + 5;
    end
    i = i + 2;

    % Second loop for 8836 iterations
    for n = 1:8836
        for j = 1:102
            Bx(nr + 1) = a * bitshift(bitand(data(i), 12), -2) + data(i+1) * 256 + data(i+2);
            By(nr + 1) = a * bitand(data(i), 3) + data(i+3) * 256 + data(i+4);
            nr = nr + 1;
            i = i + 5;
        end
        i = i + 2;
    end

    % Final loop for 82 iterations
    for j = 1:82
        Bx(nr + 1) = a * bitshift(bitand(data(i), 12), -2) + data(i+1) * 256 + data(i+2);
        By(nr + 1) = a * bitand(data(i), 3) + data(i+3) * 256 + data(i+4);
        nr = nr + 1;
        i = i + 5;
    end

    % Remove trailing zeros
    while Bx(nr) == 0 || By(nr) == 0
        nr = nr - 1;
    end

    % Adjust values
    midADC = 2^17;
    Bx(1:nr) = Bx(1:nr) - midADC;
    By(1:nr) = By(1:nr) - midADC;
end
