function [x, y, depth, albedo] = cnlos(transient_location)

% Constants
c              = 3e8;   % Speed of light (meters per second)
bin_resolution = 5*10^-3/c; % Native bin resolution for SPAD is 4 ps

% Adjustable parameters
isbackprop = 0;         % Toggle backprojection
isdiffuse  = 1;         % Toggle diffuse reflection
K          = 1;         % Downsample data to (4 ps) * 2^K = 16 ps for K = 2
snr        = 8e-1;      % SNR value
z_trim     = 0;       % Set first 600 bins to zero
load(transient_location);

width = (max(lighting(:,1)) - min(lighting(:,1)))/2;

z_offset = 0;
rect_data = reshape(gt_transient, 7,7, size(gt_transient,2));
rect_data = rect_data(:,:,1:1024);

N = size(rect_data,1);        % Spatial resolution of data
M = size(rect_data,3);        % Temporal resolution of data
range = M.*c.*bin_resolution; % Maximum range for histogram

% Downsample data to 16 picoseconds
for k = 1:K
    M = M./2;
    bin_resolution = 2*bin_resolution;
    rect_data = rect_data(:,:,1:2:end) + rect_data(:,:,2:2:end);
    z_trim = round(z_trim./2);
    z_offset = round(z_offset./2);
end

% Set first group of histogram bins to zero (to remove direct component)
rect_data(:,:,1:z_trim) = 0;

% Define NLOS blur kernel
psf = definePsf(N,M,width./range);

% Compute inverse filter of NLOS blur kernel
fpsf = fftn(psf);
if (~isbackprop)
    invpsf = conj(fpsf) ./ (abs(fpsf).^2 + 1./snr);
else
    invpsf = conj(fpsf);
end

% Define transform operators
[mtx,mtxi] = resamplingOperator(M);

% Permute data dimensions
data = permute(rect_data,[3 2 1]);

% Define volume representing voxel distance from wall
grid_z = repmat(linspace(0,1,M)',[1 N N]);

% Step 1: Scale radiometric component
if (isdiffuse)
    data = data.*(grid_z.^4);
else
    data = data.*(grid_z.^2);
end

% Step 2: Resample time axis and pad result
tdata = zeros(2.*M,2.*N,2.*N);
tmp = mtx*data(:,:);
tdata(1:end./2,1:end./2,1:end./2)  = reshape(mtx*data(:,:),[M N N]);

% Step 3: Convolve with inverse filter and unpad result
tvol = ifftn(fftn(tdata).*invpsf);
tvol = tvol(1:end./2,1:end./2,1:end./2);

% Step 4: Resample depth axis and clamp results
vol  = reshape(mtxi*tvol(:,:),[M N N]);
vol  = max(real(vol),0);




tic_z = linspace(0,range./2,size(vol,1));
tic_y = linspace(min(lighting(:,2)),max(lighting(:,2)),size(vol,2));
tic_x = linspace(min(lighting(:,1)),max(lighting(:,1)),size(vol,3));

[x, y] = meshgrid(tic_x, tic_y);

% Crop and flip reconstructed volume for visualization
%ind = round(M.*2.*width./(range./2));
ind = size(vol,1);
vol = vol(:,:,end:-1:1);
vol = vol((1:ind)+z_offset,:,:);

tic_z = tic_z((1:ind)+z_offset);

% View result

[Y, I] = max(vol,[],1);
depth = squeeze(tic_z(I));
albedo = squeeze(max(vol,[],1));
figure; 
subplot(211); imagesc(tic_x, tic_y, depth);
subplot(212);imagesc(tic_x,tic_y,albedo);

figure;
subplot(1,3,1);
imagesc(tic_x,tic_y,squeeze(max(vol,[],1)));
title('Front view');
set(gca,'XTick',linspace(min(tic_x),max(tic_x),3));
set(gca,'YTick',linspace(min(tic_y),max(tic_y),3));
xlabel('x (m)');
ylabel('y (m)');
colormap('gray');
axis square;

subplot(1,3,2);
imagesc(tic_x,tic_z,squeeze(max(vol,[],2)));
title('Top view');
set(gca,'XTick',linspace(min(tic_x),max(tic_x),3));
set(gca,'YTick',linspace(min(tic_z),max(tic_z),3));
xlabel('x (m)');
ylabel('z (m)');
colormap('gray');
axis square;

subplot(1,3,3);
imagesc(tic_z,tic_y,squeeze(max(vol,[],3))')
title('Side view');
set(gca,'XTick',linspace(min(tic_z),max(tic_z),3));
set(gca,'YTick',linspace(min(tic_y),max(tic_y),3));
xlabel('z (m)');
ylabel('y (m)');
colormap('gray');
axis square;
end

function psf = definePsf(U,V,slope)
% Local function to computeD NLOS blur kernel

x = linspace(-1,1,2.*U);
y = linspace(-1,1,2.*U);
z = linspace(0,2,2.*V);
[grid_z,grid_y,grid_x] = ndgrid(z,y,x);

% Define PSF
psf = abs(((4.*slope).^2).*(grid_x.^2 + grid_y.^2) - grid_z);
psf = double(psf == repmat(min(psf,[],1),[2.*V 1 1]));
psf = psf./sum(psf(:,U,U));
psf = psf./norm(psf(:));
psf = circshift(psf,[0 U U]);
end

function [mtx,mtxi] = resamplingOperator(M)
% Local function that defines resampling operators

mtx = sparse([],[],[],M.^2,M,M.^2);

x = 1:M.^2;
mtx(sub2ind(size(mtx),x,ceil(sqrt(x)))) = 1;
mtx  = spdiags(1./sqrt(x)',0,M.^2,M.^2)*mtx;
mtxi = mtx';

K = log(M)./log(2);
for k = 1:round(K)
    m_length = floor(size(mtx,1)/2)*2;
    mtx  = 0.5.*(mtx(1:2:m_length,:)  + mtx(2:2:m_length,:));
    mtxi = 0.5.*(mtxi(:,1:2:m_length) + mtxi(:,2:2:m_length));
end
end
