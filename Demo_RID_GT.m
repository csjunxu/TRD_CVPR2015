%--------------------------------------------------------------------------
clear;
GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_MeanImage\';
GT_fpath = fullfile(GT_Original_image_dir, '*.png');
CC_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_NoisyImage\';
CC_fpath = fullfile(CC_Original_image_dir, '*.png');
GT_im_dir  = dir(GT_fpath);
CC_im_dir  = dir(CC_fpath);
im_num = length(CC_im_dir);

method           =  'TRD';
nSig     =  [15];
format compact;

PSNR = [];
SSIM = [];
for i = 1:im_num
    IM =   im2double(imread( fullfile(CC_Original_image_dir,CC_im_dir(i).name) ));
    IM_GT = im2double(imread(fullfile(GT_Original_image_dir, GT_im_dir(i).name)));
    S = regexp(CC_im_dir(i).name, '\.', 'split');
    IMname = S{1};
    [h,w,ch] = size(IM);
    if ch==1
        IMin_y = I;
    else
        % change color space, work on illuminance only
        IMin_ycbcr = rgb2ycbcr(IM);
        IMin_y = IMin_ycbcr(:, :, 1);
        IMin_cb = IMin_ycbcr(:, :, 2);
        IMin_cr = IMin_ycbcr(:, :, 3);
    end
    %         randn('seed',0);
    %         IMin_y          =   I+ nSig*randn(size(I));
    % IMin_y = double(uint8(IMin_y));
    
    if nSig == 10
        load JointTraining_7x7_400_180x180_stage=5_sigma=10.mat;
        10
    elseif nSig == 15
        load JointTraining_7x7_400_180x180_stage=5_sigma=15.mat;
        15
    elseif nSig == 20
        load JointTraining_7x7_400_180x180_stage=5_sigma=25.mat;
        25
    elseif nSig == 50
        load JointTraining_7x7_400_180x180_stage=5_sigma=50.mat;
        50
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
    input = pad(IMin_y);
    noisy = pad(IMin_y);
    for s = 1:stage
        deImg = denoisingOneStepGMixMFs(noisy*255, input, trained_model{s});
        t = crop(deImg);
        deImg = pad(t);
        input = deImg;
    end
    x_star = max(0, min(t(:), 255));
    IMout_y = reshape(x_star,h,w);
    if ch==1
        IMout = IMout_y/255;
    else
        IMout_ycbcr = zeros(size(IM));
        IMout_ycbcr(:, :, 1) = IMout_y/255;
        IMout_ycbcr(:, :, 2) = IMin_cb;
        IMout_ycbcr(:, :, 3) = IMin_cr;
        IMout = ycbcr2rgb(IMout_ycbcr);
    end
    PSNR = [PSNR csnr( IMout*255, IM_GT*255, 0, 0 )];
    SSIM = [SSIM cal_ssim( IMout*255, IM_GT*255, 0, 0 )];
    imwrite(IMout, ['C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_' method '\' method '_' IMname '.png']);
end
mPSNR = mean(PSNR);
mSSIM = mean(SSIM);
save(['C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_', method, '_CCNoise' num2str(im_num) '.mat'],'nSig','PSNR','mPSNR','SSIM','mSSIM');