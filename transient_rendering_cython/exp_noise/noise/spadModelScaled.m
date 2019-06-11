function [transientSPAD] = spadModelScaled(transientIdeal, spadPara)

% this function implementes the SPAD modeling
% reference: A Computational Model of a Single-Photon Avalanche Diode Sensor for Transient Imaging
% Input:      transientIdeal: 1*t vector
%             spadPara:       struct stores jitter curve and back ground noise para
% Output:     transientSPAD:  1*t vector

assert(size(transientIdeal, 1) == 1);

%% SPAD Parameters
M   = spadPara.M;           % Measurements % 1e6
PDP = 1;                  % Photon Detection Probability (-)
APP = 0.01;                 % After Pulsing Probability (-)
% c   = 3e8;                  % Speed of light (m/s)
% exp = 0.005;                % Film exposition (m)
dt  = 4e-12;                % Temporal Resolution (s), 4e-12, exp / c
tHO = 1e-6;                 % Hold Off time (s), 1e-8
HO  = round(tHO / dt);      % After Pulse (#bin)
LPF = 8e7;                  % Laser Pulse Frequency (Hz, s^-1)
RR = round((1/LPF) / dt);   % Repetition Rate (#bin)


jitterCounts = spadPara.jitterCounts;
jittersAll = spadPara.jittersAll;
muNoise = spadPara.muNoise;

%% Processing and Simulation
transientSPAD = zeros(size(transientIdeal));

tEnd = size(transientIdeal, 2);  % #bin
tVec = randsample(1:tEnd, M, true, transientIdeal); % Importance sampling
jitterVec = round(randsample(jittersAll, M, true, jitterCounts) / dt);

tLast = 0;
for idx = 1 : M
    t = tVec(idx);  % #bin
    tNewStamp = t + (idx-1) * RR;    % pile-up effect
    
    jitter = jitterVec(idx);    % Time jitter
    t = t + jitter;
    
%     if (tNewStamp > tLast) && (t >= 1)...
%         && (t <= tEnd) && (rand < PDP)  % Photon detected
    if (t >= 1) && (t <= tEnd) && (rand < PDP) 
        transientSPAD(1, t) = transientSPAD(1, t) + 1;
        t = t + HO;    % Hold off time
        
        n = 1;
        while (t <= tEnd)                    
            if (rand < APP^n)  % if afterpulse
                transientSPAD(1, t) = transientSPAD(1, t) + 1;
                t = t + HO;
                n = n + 1;
            else
                break;
            end
        end
        
        tLast = t + (idx-1) * RR;
    end  % if photon detected
end

% Background noise
muNoiseBack = muNoise * M / sum(jitterCounts) * tEnd / length(jittersAll);
backgroundNoise = poissrnd(muNoiseBack, 1, tEnd);
transientSPAD = transientSPAD + backgroundNoise;


end

