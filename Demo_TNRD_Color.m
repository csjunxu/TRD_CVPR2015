%--------------------------------------------------------------------------
clear;
Original_image_dir  =    'C:\Users\csjunxu\Desktop\JunXu\Datasets\kodak24\kodak_color\';
fpath = fullfile(Original_image_dir, '*.png');
im_dir  = dir(fpath);
im_num = length(im_dir);

method           =  'TNRD';
write_sRGB_dir = ['C:/Users/csjunxu/Desktop/ICCV2017/24images/'];
if ~isdir(write_sRGB_dir)
    mkdir(write_sRGB_dir)
end
format compact;

% nSig = [40 20 30];
% nSig = [30 10 50];
nSig = [5 30 15];

modelnSig = [10 35 15];
PSNR = [];
SSIM = [];
for i = 1:im_num
    IM_GT = double(imread(fullfile(Original_image_dir, im_dir(i).name)));
    S = regexp(im_dir(i).name, '\.', 'split');
    IMname = S{1};
    [h,w,ch] = size(IM_GT);
    fprintf('%s: \n',im_dir(i).name);
    IMin = zeros(size(IM_GT));
    for c = 1:ch
        randn('seed',0);
        IMin(:, :, c) = IM_GT(:, :, c) + nSig(c) * randn(size(IM_GT(:, :, c)));
    end
    fprintf('The initial PSNR = %2.4f, SSIM = %2.4f. \n', csnr( IMin,IM_GT, 0, 0 ), cal_ssim( IMin, IM_GT, 0, 0 ));
    IMout = zeros(size(IM_GT));
    for c = 1:ch
        
        if modelnSig(c) == 10
            load JointTraining_7x7_400_180x180_stage=5_sigma=10.mat;
            10
        elseif modelnSig(c) == 15
            load JointTraining_7x7_400_180x180_stage=5_sigma=15.mat;
            15
        elseif modelnSig(c) == 25
            load JointTraining_7x7_400_180x180_stage=5_sigma=25.mat;
            25
        elseif modelnSig(c) == 35
            load JointTraining_7x7_400_180x180_stage=5_sigma=35.mat;
            35
        elseif modelnSig(c) == 50
            load JointTraining_7x7_400_180x180_stage=5_sigma=50.mat;
            50
        elseif modelnSig(c) == 75
            load JointTraining_7x7_400_180x180_stage=5_sigma=75.mat;
            75
        end
        %% default setting
        filter_size = 7;
        m = filter_size^2 - 1;
        filter_num = 48;
        BASIS = gen_dct2(filter_size);
        BASIS = BASIS(:,2:end);
        %% pad and crop operation
        bsz = filter_size+1;
        bndry = [bsz,bsz];
        pad   = @(x) padarray(x,bndry,'symmetric','both');
        crop  = @(x) x(1+bndry(1):end-bndry(1),1+bndry(2):end-bndry(2));
        %% MFs means and precisions
        KernelPara.fsz = filter_size;
        KernelPara.filtN = filter_num;
        KernelPara.basis = BASIS;
        trained_model = save_trained_model(cof, MFS, stage, KernelPara);
        input = pad(IMin(:,:,c));
        noisy = pad(IMin(:,:,c));
        for s = 1:stage
            deImg = denoisingOneStepGMixMFs(noisy, input, trained_model{s});
            t = crop(deImg);
            deImg = pad(t);
            input = deImg;
        end
        x_star = max(0, min(t(:), 255));
        IMoutcc = reshape(x_star,h,w);
        IMout(:,:,c) = IMoutcc;
    end
    PSNR = [PSNR csnr( IMout, IM_GT, 0, 0 )];
    SSIM = [SSIM cal_ssim( IMout, IM_GT, 0, 0 )];
    fprintf('The final PSNR = %2.4f, SSIM = %2.4f. \n', PSNR(end), SSIM(end));
    imwrite(IMout/255, [write_sRGB_dir method '_nSig' num2str(nSig(1)) num2str(nSig(2)) num2str(nSig(3)) '_modelnSig' num2str(modelnSig(1)) num2str(modelnSig(2)) num2str(modelnSig(3)) '_' IMname '.png']);
end
mPSNR = mean(PSNR);
mSSIM = mean(SSIM);
save([write_sRGB_dir method, '_nSig' num2str(nSig(1)) num2str(nSig(2)) num2str(nSig(3)) '_modelnSig' num2str(modelnSig(1)) num2str(modelnSig(2)) num2str(modelnSig(3)) '.mat'],'nSig','PSNR','mPSNR','SSIM','mSSIM');


nSig = [30 10 50];

modelnSig = [35 10 50];
PSNR = [];
SSIM = [];
for i = 1:im_num
    IM_GT = double(imread(fullfile(Original_image_dir, im_dir(i).name)));
    S = regexp(im_dir(i).name, '\.', 'split');
    IMname = S{1};
    [h,w,ch] = size(IM_GT);
    fprintf('%s: \n', im_dir(i).name);
    IMin = zeros(size(IM_GT));
    for c = 1:ch
        randn('seed',0);
        IMin(:, :, c) = IM_GT(:, :, c) + nSig(c) * randn(size(IM_GT(:, :, c)));
    end
    fprintf('The initial PSNR = %2.4f, SSIM = %2.4f. \n', csnr( IMin,IM_GT, 0, 0 ), cal_ssim( IMin, IM_GT, 0, 0 ));
    IMout = zeros(size(IM_GT));
    for c = 1:ch
        
        if modelnSig(c) == 10
            load JointTraining_7x7_400_180x180_stage=5_sigma=10.mat;
            10
        elseif modelnSig(c) == 15
            load JointTraining_7x7_400_180x180_stage=5_sigma=15.mat;
            15
        elseif modelnSig(c) == 25
            load JointTraining_7x7_400_180x180_stage=5_sigma=25.mat;
            25
        elseif modelnSig(c) == 35
            load JointTraining_7x7_400_180x180_stage=5_sigma=35.mat;
            35
        elseif modelnSig(c) == 50
            load JointTraining_7x7_400_180x180_stage=5_sigma=50.mat;
            50
        elseif modelnSig(c) == 75
            load JointTraining_7x7_400_180x180_stage=5_sigma=75.mat;
            75
        end
        %% default setting
        filter_size = 7;
        m = filter_size^2 - 1;
        filter_num = 48;
        BASIS = gen_dct2(filter_size);
        BASIS = BASIS(:,2:end);
        %% pad and crop operation
        bsz = filter_size+1;
        bndry = [bsz,bsz];
        pad   = @(x) padarray(x,bndry,'symmetric','both');
        crop  = @(x) x(1+bndry(1):end-bndry(1),1+bndry(2):end-bndry(2));
        %% MFs means and precisions
        KernelPara.fsz = filter_size;
        KernelPara.filtN = filter_num;
        KernelPara.basis = BASIS;
        trained_model = save_trained_model(cof, MFS, stage, KernelPara);
        input = pad(IMin(:,:,c));
        noisy = pad(IMin(:,:,c));
        for s = 1:stage
            deImg = denoisingOneStepGMixMFs(noisy, input, trained_model{s});
            t = crop(deImg);
            deImg = pad(t);
            input = deImg;
        end
        x_star = max(0, min(t(:), 255));
        IMoutcc = reshape(x_star,h,w);
        IMout(:,:,c) = IMoutcc;
    end
    PSNR = [PSNR csnr( IMout, IM_GT, 0, 0 )];
    SSIM = [SSIM cal_ssim( IMout, IM_GT, 0, 0 )];
    fprintf('The final PSNR = %2.4f, SSIM = %2.4f. \n', PSNR(end), SSIM(end));
    imwrite(IMout/255, [write_sRGB_dir method '_nSig' num2str(nSig(1)) num2str(nSig(2)) num2str(nSig(3)) '_modelnSig' num2str(modelnSig(1)) num2str(modelnSig(2)) num2str(modelnSig(3)) '_' IMname '.png']);
end
mPSNR = mean(PSNR);
mSSIM = mean(SSIM);
save([write_sRGB_dir method, '_nSig' num2str(nSig(1)) num2str(nSig(2)) num2str(nSig(3)) '_modelnSig' num2str(modelnSig(1)) num2str(modelnSig(2)) num2str(modelnSig(3)) '.mat'],'nSig','PSNR','mPSNR','SSIM','mSSIM');
