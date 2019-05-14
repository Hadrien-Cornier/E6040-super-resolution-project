%%% K-space truncation %%%
%NOTE: We let z=slice, so data here is in the order of (z, x, y)
% change to your root directory
rootdir='G:\E6040\superresolution\HCP_1200';
% rootdir='D:\HCP_1200';
fileFolder=fullfile(rootdir);
datadir=dir(fullfile(fileFolder)); 
h = waitbar(0,'please wait..');
for i=1:length(datadir)-2
    cd (rootdir);
    sub_id=datadir(i+2).name;
    cd ([rootdir,'\',sub_id,'\unprocessed\3T\T1w_MPR1\']);
    obj=load_untouch_nii('*.nii.gz');
    Data=obj.img;
    Data_float=double(Data);
    Data_norm=Data_float./4095.0;
    %% 3DFFT
    kData = fftshift(fftn(ifftshift(Data_norm)));
    % F= squeeze(kData(:,:,50));
    % F = abs(F); % Get the magnitude
    % F = mat2gray(F); % Use mat2gray to scale the image between 0 and 1
    % figure,imshow(F,[]);
    
    Factor_truncate=2.5;
    x_range=round(size(kData,2)/Factor_truncate);
    y_range=round(size(kData,3)/Factor_truncate);
    kData_truncate=kData;
    kData_truncate(:,1:x_range,:)=0;
    kData_truncate(:,:,1:y_range)=0;
    kData_truncate(:,end-x_range+1:end,:)=0;
    kData_truncate(:,:,end-y_range+1:end)=0;   
    
    % D= squeeze(kData_truncate(50,:,:));
    % D = abs(D); % Get the magnitude
    % D = mat2gray(D); % Use mat2gray to scale the image between 0 and 1
    % figure,imshow(D,[]);
    
%% zeroCrop
%     % up to 3D, crop out edge zeros
%     data=kData_truncate;
%     zMin = find(any(any(data(:,:,:,1),2),3),1,'first');
%     zMax = find(any(any(data(:,:,:,1),2),3),1,'last');
% 
%     xMin = find(squeeze(any(any(data(:,:,:,1),1),3)),1,'first');
%     xMax = find(squeeze(any(any(data(:,:,:,1),1),3)),1,'last');
% 
%     yMin = find(squeeze(any(any(data(:,:,:,1),1),2)),1,'first');
%     yMax = find(squeeze(any(any(data(:,:,:,1),1),2)),1,'last');
% 
%     szData = size(data);
%     kout = data(zMin:zMax,xMin:xMax,yMin:yMax,:);
% 
%     szOut = size(kout);
%     kout = reshape(kout,[szOut(1:3) szData(4:end)]);
    
    %% 3DIFFT
    kout = kData_truncate;
    out = fftshift(ifftn(ifftshift(kout)));
    out_real =abs(out);
    nz=size(Data_norm,1);
    nx=size(Data_norm,2);
    ny=size(Data_norm,3);
    out_norm=out_real./max(out_real(:));
    for j=1:nz
        out_float(j,:,:) = imresize(squeeze(out_norm(j,:,:)), [nx ny] ,'bilinear');
    end
    
    out_final=round(out_float.*4095.0);
    out_final=int16(out_final);
    save([sub_id, '_LR.mat'],'out_final');
    waitbar(i/(length(datadir)-2),h,[num2str(i),'/',num2str(length(datadir)-2)])
end
close(h)

% S= squeeze(Data_norm(120,:,:));
% figure,imshow(S,[]);
%     
% F= squeeze(out_float(120,:,:));
% figure,imshow(F,[]);