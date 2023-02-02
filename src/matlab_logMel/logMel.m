function [mel] = logMel(filepath)
[audioIn, fs] = audioread(filepath);
new_sr = 16000;
window = round(0.025*new_sr);
shift = round(0.010*new_sr);
audioIn = resample(audioIn,new_sr,fs);

mel = melSpectrogram(audioIn,new_sr, ...
                   'Window',hamming(window,'periodic'), ...
                   'OverlapLength',shift, ...
                   'NumBands',40);
mel = 20*log10(mel);
mel = transpose(mel);
end