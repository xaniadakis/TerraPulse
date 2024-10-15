function [calibrated_Bx, calibrated_By] = calibrate_HYL(Bx, By, length)
    a1_mVnT = 55.0;
    a2_mVnT = 55.0;

    a1 = a1_mVnT * 1e-3 / 1e3;
    a2 = a2_mVnT * 1e-3 / 1e3;
    ku = 4.26;
    c1 = a1 * ku;
    c2 = a2 * ku;
    d = 2^18;
    V = 4.096 * 2;

    scale1 = c1 * d / V;
    scale2 = c2 * d / V;

    calibrated_Bx = -Bx(1:length) / scale1;
    calibrated_By = -By(1:length) / scale2;
end