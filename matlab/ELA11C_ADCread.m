function [Bx, By, nr, tini] = ELA11C_ADCread(fn)
    K = 65536;
    fid = fopen(fn, 'rb');
    header = fread(fid, 64, 'uint8');

    first_sample = header(49) * 256 + header(50);
    tini = first_sample / 1250e3;
    tini = tini - 20e-6;

    data = fread(fid, inf, 'uint8');
    fclose(fid);

    ld = length(data) - 64;
    Bx = zeros(floor(ld/4), 1);
    By = zeros(floor(ld/4), 1);

    nr = 1;
    i = 2;
    
    for j = 1:89
        Bx(nr) = K * bitshift(bitand(data(i), 12), -2) + data(i+1) * 256 + data(i+2);
        By(nr) = K * bitand(data(i), 3) + data(i+3) * 256 + data(i+4);

        nr = nr + 1;
        i = i + 5;
    end
    
    i = i + 2;

    for n = 1:8836
        for j = 1:102
            Bx(nr) = K * bitshift(bitand(data(i), 12), -2) + data(i+1) * 256 + data(i+2);
            By(nr) = K * bitand(data(i), 3) + data(i+3) * 256 + data(i+4);

            nr = nr + 1;
            i = i + 5;
        end
        i = i + 2;
    end

    for j = 1:82
        Bx(nr) = K * bitshift(bitand(data(i), 12), -2) + data(i+1) * 256 + data(i+2);
        By(nr) = K * bitand(data(i), 3) + data(i+3) * 256 + data(i+4);

        nr = nr + 1;
        i = i + 5;
    end

    while Bx(nr) == 0 || By(nr) == 0
        nr = nr - 1;
    end

    fprintf('Number of samples %d (fs=3kHz->901442|+1)\n', nr);

    midADC = 2^18 / 2;
    Bx = Bx(1:nr) - midADC;
    By = By(1:nr) - midADC;
end
