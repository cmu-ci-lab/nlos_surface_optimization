clear; clc; close all;

load('jitter');
counts_1 = counts_1/sum(counts_1);

bin = 4*10^-12;

figure;
plot(t_1/bin, counts_1);


interval = t_1(2)-t_1(1);
sigma = 1 * bin/2.355 /interval;


kernal_size = round(8*sigma);
if mod(kernal_size,2) == 0
    kernal_size = kernal_size+1;
end

mid = (kernal_size+1)/2;
x = 1:kernal_size;
g = 1/sqrt(2*pi)/sigma*exp(- (x - mid).^2/2/sigma/sigma);

figure;
plot(x, g, 'r');

figure;
y = filter(g,1,counts_1);
new_t = t_1(1:end-mid+1)/bin;
w = y(mid:end);
plot(new_t, w);

scale = 1;
t = round(new_t/scale);
x = unique(t);
new_w = zeros(length(x),1);
for i = 1:length(x)
    new_w(i) = sum(w(t==x(i)));
end
figure;
plot(x*scale,new_w);

[B,I] = sort(new_w, 'descend');

B = B(1:40);
I = I(1:40);
hold on;
plot(x(I)*scale, B, 'r.');


g1 = new_w(2:end)-new_w(1:end-1);
g = ([g1; 0] + [0; g1])/2;

figure;
plot(x*scale,g);


time = x(I);
[jitter_time, Y] = sort(x(I));
jitter_weight = B(Y);
grad = g(I);
jitter_grad = grad(Y);

figure
plot(jitter_time, jitter_weight, 'r.');
figure;
plot(jitter_time, jitter_grad, 'r.');

jitter_offset = find(jitter_time == 0) - 1;
save('jitter_info', 'jitter_time', 'jitter_weight', 'jitter_grad', 'jitter_offset');

