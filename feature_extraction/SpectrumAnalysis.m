
% SpectrumAnalysis Ƶ�׷���
% need MATLAB2019a or MATLAB2019B

clear   % ���������
clc     % ��������д���
tic     % ��ʱ
main_dir = 'D:\bargainingPower\01_MatlabCode\04_spectrum\Spectrum\Bern Barcelona\ALL\';
% for sd = 1:length(main_dir)
cur_dir = dir([main_dir,'\*.txt']);         % �г��ļ�������
chan1 = zeros(length(cur_dir),6);           % Ԥ���ڴ�
chan2 = zeros(length(cur_dir),6);           % Ԥ���ڴ�
for cd = 1:length(cur_dir)
    Raw  = load([main_dir,cur_dir(cd).name]);     % ����txt 
    [SamplePot,~] = size(Raw);                    % ����
    Sample1 = zeros(1,SamplePot);                  % Ԥ���ڴ�
    Sample2 = zeros(1,SamplePot);                  % Ԥ���ڴ�
for i=1:SamplePot
        Sample1(i) = Raw(i,1);      % ������ N = fs * ts;
        Sample2(i) = Raw(i,2);
end
data1 = Sample1;
data2 = Sample2;
fs = 512;                        % ����Ƶ�� Hz �������� 1/fs
        % 1 Ƶ�����ʱ�
        % p = bandpower(x,fs,freqrange) returns the average power in the
        % frequency range, freqrange, specified as a two-element vector.
        % You must input the sample rate, fs, to return the power in a
        % specified frequency range. bandpower uses a modified periodogram
        % to determine the average power in freqrange.
        % x - �����ź�
        % fs - ����Ƶ��
        % freqrange - Ƶ�ʷ�Χ
        pband = bandpower(data1,fs,[60 140]);
        ptot = bandpower(data1,fs,[0 60]);
        chan1(cd,1) = 100*(pband/ptot);
        pband = bandpower(data2,fs,[60 140]);
        ptot = bandpower(data2,fs,[0 60]);
        chan2(cd,1) = 100*(pband/ptot);
        
        % 2 ���������ܶ�  power spectrum density
        chan1(cd,2) =max((1/(fs*SamplePot)) * abs(fft(data1)).^2);
        chan2(cd,2) =max((1/(fs*SamplePot)) * abs(fft(data2)).^2);
        
        % 3  ������
        [margin] = Spectrum_1(data1);
        chan1(cd,3) = max(margin);
        [margin] = Spectrum_1(data2);
        chan2(cd,3) = max(margin);
        
        % 4 Ƶ������ 
        % Transpose - ת�� 
        % spectralCentroid - Spectral centroid in Hz,
        % returned as a scalar, vector, or matrix. Each row of centroid
        % corresponds to the spectral centroid of a window of x. Each
        % column of centroid corresponds to an independent channel.
       chan1(cd,4) = max(transpose(spectralCentroid(transpose(data1(1,:)),fs)));
       chan2(cd,4) = max(transpose(spectralCentroid(transpose(data2(1,:)),fs)));
    
        % 5 Ƶ�׷��
        chan1(cd,5) = spectralkurtosis(data1,fs);
        chan2(cd,5) = spectralkurtosis(data2,fs);
        
        % 6 Ƶ���� Spectral entropy
        chan1(cd,6) = max(pentropy(data1,fs)) ;   
        chan2(cd,6) = max(pentropy(data2,fs)) ; 
end
        aver = (chan1 + chan2)/2;
        writematrix(aver, 'Frequency_features.csv')     % ����Ϊ.csv
        toc                                  % ��ʱ





