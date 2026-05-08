%%读取Mstar图像复数据，并取子带子孔径
clear;
ReadPath = 'D:\第一师范工作\科研\数据\MSTAR\MSTAR-PublicTargetChips-T72-BMP2-BTR70-SLICY\MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY\TARGETS\TRAIN\17_DEG\BTR70\SN_C71\';
SavePath = 'D:\第一师范工作\科研\数据\MstarDatasetSub\17_DEG\BTR_60\';
FileType = '*.004';

Files_read = dir([ReadPath FileType]);
NumberOfFiles = length(Files_read);
my_num=NumberOfFiles;



for k=50%:my_num
    clut_file=Files_read(k).name;
    s2=strcat(ReadPath,clut_file(1,:));
    FIDread = fopen(s2,'rb','ieee-be');
    ImgColumns = 0;
    ImgRows = 0;
    
    while ~feof(FIDread)                                % 在PhoenixHeader找到图片尺寸大小
        Text = fgetl(FIDread);
        if ~isempty(strfind(Text,'NumberOfColumns'))
            ImgColumns = str2double(Text(18:end));
            Text = fgetl(FIDread);
            ImgRows = str2double(Text(15:end));
            Text = fgetl(FIDread);
            TargetType = Text(13:end);
            Text = fgetl(FIDread);
            Text = fgetl(FIDread);
            TargetAz = str2double(Text(10:end));
            break;
        end
    end
    while ~feof(FIDread)                                 % 跳过PhoenixHeader
        Text = fgetl(FIDread);
        if ~isempty(strfind(Text,'[EndofPhoenixHeader]'))
            break
        end
    end
    Mag = fread(FIDread,ImgColumns*ImgRows,'float32','ieee-be');
    Phase = fread(FIDread,ImgColumns*ImgRows,'float32','ieee-be');
    Img1 = reshape(Mag,[ImgColumns ImgRows]);
    Img2 = reshape(Phase,[ImgColumns ImgRows]);
    Img=Img1.*exp(1j*Img2);
    CenterN_Column = floor(ImgColumns/2);
    CenterN_Row = floor(ImgRows/2);
    ImgOut = Img((CenterN_Column-30):(CenterN_Column+29),(CenterN_Row-30):(CenterN_Row+29));
    ImgOut_f= fftshift(fft2(ImgOut));
        figure;imagesc(abs(ImgOut));
        figure;imagesc(abs(ImgOut_f));
    
    p(1,:) = 8:30;
    p(2,:) = 20:42;
    p(3,:) = 31:53;
    
    q(1,:) = 5:30;
    q(2,:) = 20:45;
    q(3,:) = 30:55;
    
    for m=1:3
        for n=1:3
            filter = zeros(60,60);
            filter(p(m,:),q(n,:))=1;
            ImagOut_f_sub = ImgOut_f.*filter;
            ImagOut_sub = ifft2(ImagOut_f_sub);
            figure;imagesc(abs(ImagOut_f_sub));
            figure;imagesc(abs(ImagOut_sub));
            
%             FolderName_R = [SavePath '\sub' num2str(m) '_' num2str(n) '\Real' ];
%             if ~exist(FolderName_R, 'dir')
%                 mkdir(FolderName_R);
%             end
%             FolderName_I = [SavePath '\sub' num2str(m) '_' num2str(n) '\Imag' ];
%             if ~exist(FolderName_I, 'dir')
%                 mkdir(FolderName_I);
%             end
%             FileSaveName_R = [TargetType '_sub' num2str(m) '_' num2str(n) '_R_' num2str(k) '.txt'];
%             FileSaveName_I = [TargetType '_sub' num2str(m) '_' num2str(n) '_I_' num2str(k) '.txt'];
%             Files_save_R = [FolderName_R '\' FileSaveName_R];
%             Files_save_I = [FolderName_I '\' FileSaveName_I];
%             dlmwrite(Files_save_R, real(ImagOut_sub));
%             dlmwrite(Files_save_I, imag(ImagOut_sub));
        end
    end
    fclose (FIDread);
    
end




