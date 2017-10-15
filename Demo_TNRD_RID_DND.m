clear;
Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2018 Denoising\dnd_2017\images_srgb\';
fpath = fullfile(Original_image_dir, '*.mat');
im_dir  = dir(fpath);
im_num = length(im_dir);
load 'C:\Users\csjunxu\Desktop\CVPR2018 Denoising\dnd_2017\info.mat';

method = 'TNRD';
% write image directory
write_MAT_dir = ['C:/Users/csjunxu/Desktop/CVPR2018 Denoising/dnd_2017Results/'];
write_sRGB_dir = ['C:/Users/csjunxu/Desktop/CVPR2018 Denoising/dnd_2017Results/' method];
if ~isdir(write_sRGB_dir)
    mkdir(write_sRGB_dir)
end

for nSig     =  [ 10 ]
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
    PSNR = [];
    SSIM = [];
    nPSNR = [];
    nSSIM = [];
    RunTime = [];
    for i = 1:im_num
        load(fullfile(Original_image_dir, im_dir(i).name));
        S = regexp(im_dir(i).name, '\.', 'split');
        [h,w,ch] = size(InoisySRGB);
        for j = 1:size(info(1).boundingboxes,1)
            time0 = clock;
            IMinname = [S{1} '_' num2str(j)];
            IMin = 255*InoisySRGB(info(i).boundingboxes(j,1):info(i).boundingboxes(j,3),info(i).boundingboxes(j,2):info(i).boundingboxes(j,4),1:3);
            IM_GT = IMin;
            IMout = zeros(size(IMin));
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
            imwrite(IMou/255, [write_sRGB_dir '/' method '_our_' IMname '.png']);
        end
    end
    mPSNR = mean(PSNR);
    mSSIM = mean(SSIM);
    mnPSNR = mean(nPSNR);
    mnSSIM = mean(nSSIM);
    mRunTime = mean(RunTime);
    matname = sprintf([write_MAT_dir method '_DND.mat']);
    save(matname,'PSNR','SSIM','mPSNR','mSSIM','nPSNR','nSSIM','mnPSNR','mnSSIM','RunTime','mRunTime');
end

