function [logmel] = logMel(filepath)
    [audioIn, fs] = audioread(filepath);
    new_sr = 16000;
    window = round(0.025*new_sr);
    shift = round(0.015*new_sr);
    audioIn = resample(audioIn,new_sr,fs);

    melS = melSpectrogram(audioIn, new_sr, ...
                       'Window',hann(window,"periodic"), ...
                       'OverlapLength',shift, ...
                       'NumBands', 40);

    % Decibels
    logmel = 20*log10(melS + eps);

    % Normalize coefficients
    logmel = logmel - mean(logmel,2);
    logmel = logmel./std(logmel,[],2);
end