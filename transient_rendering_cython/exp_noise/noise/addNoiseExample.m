clear; clc; close all;

load('../setup/bunny_transient');

% controls level of ambient noise
muNoise = 10000;

% controls number of measured photons
M = 2e4;
load('jitter.mat');
counts = counts_1;% / sum(counts_1);
spadPara.jitterCounts = counts;
spadPara.jittersAll = t_1;

spadPara.muNoise = muNoise;
spadPara.M = M;

%transientNoise =...
%    spadModel(transient, spadPara) / M * sum(transient);
transientScaled = zeros(size(gt_transient));
for i = 1:size(gt_transient,1)
transientScaled(i,:) =...
    spadModelScaled(gt_transient(i,:), spadPara) / M * sum(transient(i,:));
end

% figure; 

% plot(time, transientNoise, '-k');
% hold on;

%plot(time, transientScaled, '-r');
%hold on;

%plot(time, transient, '-b');
%hold off;