%%读取Mstar图像复数据

clear;
ReadPath = 'D:\第一师范工作\科研\数据\MSTAR\MSTAR_PUBLIC_T_72_VARIANTS_CD1\15_DEG\COL2\SCENE1\A64\';
SavePath = 'D:\第一师范工作\科研\数据\MstarDataset\t72_15DEG\';
FileType = '*.024';
title = 'A64';

Files_read = dir([ReadPath FileType]);
NumberOfFiles = length(Files_read);
my_num=NumberOfFiles;



for k=1:my_num
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
    ImgOut = Img((CenterN_Column-50):(CenterN_Column+49),(CenterN_Row-50):(CenterN_Row+49));
   
fclose (FIDread);
FileSaveName_R = [title TargetType '_R_' num2str(k) '.txt'];
FileSaveName_I = [title TargetType '_I_' num2str(k) '.txt'];
FileSaveName_Az =  [title TargetType '_Az' '.txt'];
Files_save_R = [SavePath 'Real\' FileSaveName_R];
Files_save_I = [SavePath 'Imag\' FileSaveName_I];
File_Save_AZ = [SavePath FileSaveName_Az];


dlmwrite(Files_save_R, real(ImgOut));
dlmwrite(Files_save_I, imag(ImgOut));
dlmwrite(File_Save_AZ, TargetAz,'-append');


end

 


