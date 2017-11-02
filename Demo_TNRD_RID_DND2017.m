clear;

Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2018 Denoising\dnd_2017\images_srgb\';
fpath = fullfile(Original_image_dir, '*.mat');
im_dir  = dir(fpath);
im_num = length(im_dir);
load 'C:\Users\csjunxu\Desktop\CVPR2018 Denoising\dnd_2017\info.mat';

method = 'TNRD';
write_MAT_dir = ['C:/Users/csjunxu/Desktop/CVPR2018 Denoising/dnd_2017Results/'];
write_sRGB_dir = [write_MAT_dir method];
if ~isdir(write_sRGB_dir)
    mkdir(write_sRGB_dir)
end

nSig = 10;
load JointTraining_7x7_400_180x180_stage=5_sigma=10.mat;

PSNR = [];
SSIM = [];
nPSNR = [];
nSSIM = [];
RunTime = [];
for i = 1:im_num
    load(fullfile(Original_image_dir, im_dir(i).name));
    S = regexp(im_dir(i).name, '\.', 'split');
%     [h,w,ch] = size(InoisySRGB);
    for j = 1:size(info(1).boundingboxes,1)
        time0 = clock;
        bb = info(i).boundingboxes(j,:);
        IMname = [S{1} '_' num2str(j)];
        
        %  IMin = 255*InoisySRGB(bb(1):bb(3), bb(2):bb(4),1:3);
        IMin = double(imread([Original_image_dir method '_DND_' IMname '.png']));
        [h,w,ch] = size(IMin);
        
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
        fprintf('Total elapsed time = %4.2f s\n', RunTime(end) );
        PSNR = [PSNR csnr( IMout, IM_GT, 0, 0 )];
        SSIM = [SSIM cal_ssim( IMout, IM_GT, 0, 0 )];
        fprintf('The final PSNR = %2.4f, SSIM = %2.4f. \n', PSNR(end), SSIM(end));
        imwrite(IMout/255, [write_sRGB_dir '/' method '_DND_' IMname '.png']);
    end
end

