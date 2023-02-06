filename = "C:\Users\Elmeri\PycharmProjects\Automatic-syllable_reg_DSP-NN\src\resources\audio\french\french_0.wav";

[y, Fs] = audioread(filename); %load the wav
m = audioinfo(filename); %get the wav information
audioIn = resample(y,16000,Fs);
dur = length(y)/Fs; %should give you the length in seconds

mel = logMel("C:\Users\Elmeri\PycharmProjects\Automatic-syllable_reg_DSP-NN\src\resources\audio\french\french_0.wav");