%{
filename = "C:\Users\Elmeri\PycharmProjects\Automatic-syllable_reg_DSP-NN\src\resources\audio\french\french_0.wav";

[y, Fs] = audioread(filename); %load the wav
m = audioinfo(filename); %get the wav information
audioIn = resample(y,16000,Fs);
dur = length(y)/Fs; %should give you the length in seconds

mel = logMel("C:\Users\Elmeri\PycharmProjects\Automatic-syllable_reg_DSP-NN\src\resources\audio\french\french_0.wav");
%}
%{
english = load("english_tensordata.mat");
estonian = load("estonian_tensordata.mat");

english_tensor = english.tensor;
estonian_tensor = estonian.tensor;
%}
%{
list_of_eng_LM = [];
logMel_long_eng = [];
for k = 1:length(english_tensor(:,:,:))
    logMel = squeeze(english_tensor(k,:,:));
    logMel_long_eng = cat(1,logMel_long_eng,logMel);
        if(rem(k, 100) == 0)
        fprintf("%u \n",k);
        end
end
%}
eng = squeeze(english_tensor(129,:,:));
est = squeeze(estonian_tensor(128,:,:));

english_tensor_noninf = english_tensor(english_tensor ~= -inf);

maxEng = max(unique(nonzeros(english_tensor_noninf(:))));
minEng = min(unique(nonzeros(english_tensor_noninf(:))));

maxEst = max(unique(nonzeros(estonian_tensor(:))));
minEst = min(unique(nonzeros(estonian_tensor(:))));