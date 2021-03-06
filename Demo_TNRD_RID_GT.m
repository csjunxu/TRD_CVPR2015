%--------------------------------------------------------------------------
clear;
GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2018 Denoising\cc_Results\Real_ccnoise_denoised_part\';
GT_fpath = fullfile(GT_Original_image_dir, '*mean.png');
TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2018 Denoising\cc_Results\Real_ccnoise_denoised_part\';
TT_fpath = fullfile(TT_Original_image_dir, '*real.png');
% GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_MeanImage\';
% GT_fpath = fullfile(GT_Original_image_dir, '*.png');
% TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_NoisyImage\';
% TT_fpath = fullfile(TT_Original_image_dir, '*.png');
% GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\our_Results\Real_MeanImage\';
% GT_fpath = fullfile(GT_Original_image_dir, '*.JPG');
% TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\our_Results\Real_NoisyImage\';
% TT_fpath = fullfile(TT_Original_image_dir, '*.JPG');
% GT_Original_image_dir = 'C:/Users/csjunxu/Desktop/CVPR2018 Denoising/RealisticImage/';
% GT_fpath = fullfile(GT_Original_image_dir, '*mean.JPG');
% TT_Original_image_dir = 'C:/Users/csjunxu/Desktop/CVPR2018 Denoising/RealisticImage/';
% TT_fpath = fullfile(TT_Original_image_dir, '*real.JPG');

GT_im_dir  = dir(GT_fpath);
TT_im_dir  = dir(TT_fpath);
im_num = length(TT_im_dir);

method           =  'TNRD';
write_MAT_dir = ['C:/Users/csjunxu/Desktop/CVPR2018 Denoising/cc_Results/'];
write_sRGB_dir =  ['C:/Users/csjunxu/Desktop/CVPR2018 Denoising/cc_Results/' method];
if ~isdir(write_sRGB_dir)
    mkdir(write_sRGB_dir)
end

for nSig     =  [ 25 35 50 75]
    PSNR = [];
    SSIM = [];
    nPSNR = [];
    nSSIM = [];
    RunTime = [];
    for i = 1:im_num
        IMin = double(imread(fullfile(TT_Original_image_dir,TT_im_dir(i).name) ));
        IM_GT = double(imread(fullfile(GT_Original_image_dir, GT_im_dir(i).name)));
        % S = regexp(TT_im_dir(i).name, '\.', 'split');
        IMname = TT_im_dir(i).name(1:end-9);
        [h,w,ch] = size(IMin);
        fprintf('%s: \n',TT_im_dir(i).name);
        nPSNR = [nPSNR csnr( IMin,IM_GT, 0, 0 )];
        nSSIM = [nSSIM cal_ssim( IMin, IM_GT, 0, 0 )];
        fprintf('The initial PSNR = %2.4f, SSIM = %2.4f. \n', nPSNR(end), nSSIM(end));
        IMout = zeros(size(IMin));
        time0 = clock;
        if nSig == 10
            load JointTraining_7x7_400_180x180_stage=5_sigma=10.mat;
            10
        elseif nSig == 15
            load JointTraining_7x7_400_180x180_stage=5_sigma=15.mat;
            15
        elseif nSig == 25
            load JointTraining_7x7_400_180x180_stage=5_sigma=25.mat;
            25
        elseif nSig == 35
            load JointTraining_7x7_400_180x180_stage=5_sigma=35.mat;
            35
        elseif nSig == 50
            load JointTraining_7x7_400_180x180_stage=5_sigma=50.mat;
            50
        elseif nSig == 75
            load JointTraining_7x7_400_180x180_stage=5_sigma=75.mat;
            75
        end
        for cc = 1:ch
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
            input = pad(IMin(:,:,cc));
            noisy = pad(IMin(:,:,cc));
            for s = 1:stage
                deImg = denoisingOneStepGMixMFs(noisy, input, trained_model{s});
                t = crop(deImg);
                deImg = pad(t);
                input = deImg;
            end
            x_star = max(0, min(t(:), 255));
            IMoutcc = reshape(x_star,h,w);
            IMout(:,:,cc) = IMoutcc;
        end
        RunTime = [RunTime etime(clock,time0)];
        fprintf('Total elapsed time = %f s\n', (etime(clock,time0)) );
        PSNR = [PSNR csnr( IMout, IM_GT, 0, 0 )];
        SSIM = [SSIM cal_ssim( IMout, IM_GT, 0, 0 )];
        fprintf('The final PSNR = %2.4f, SSIM = %2.4f. \n', PSNR(end), SSIM(end));
        imwrite(IMout/255, [write_sRGB_dir '/' method '_cc_' num2str(nSig) '_' IMname '.png']);
    end
    mPSNR = mean(PSNR);
    mSSIM = mean(SSIM);
    mnPSNR = mean(nPSNR);
    mnSSIM = mean(nSSIM);
    mRunTime = mean(RunTime);
    save([write_MAT_dir method '_cc' num2str(nSig) '.mat'],'nSig','PSNR','mPSNR','SSIM','mSSIM','nPSNR','mnPSNR','nSSIM','mnSSIM','RunTime','mRunTime');
end

