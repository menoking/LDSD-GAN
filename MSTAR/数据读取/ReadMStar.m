clear
ReadPath = 'D:\Document\Graduation Project\MSTAR\DAA7B02AA\TARGETS\TEST\15_DEG\T72\SN_S7\';
SavePath = 'D:\Document\Graduation Project\MSTAR\DAA7B02AA\TARGETSJPG\TEST\15_DEG\T72\SN_S7\';
FileType = '*.017';
Files = dir([ReadPath FileType]);
NumberOfFiles = length(Files);
for i = 1 : NumberOfFiles
    FileName = Files(i).name;
    NameLength = length(FileName);
    FID = fopen([ReadPath FileName],'rb','ieee-be');
    ImgColumns = 0;
    ImgRows = 0;
    while ~feof(FID)                                % 在PhoenixHeader找到图片尺寸大小
        Text = fgetl(FID);
        if ~isempty(strfind(Text,'NumberOfColumns'))
            ImgColumns = str2double(Text(18:end));
            Text = fgetl(FID);
            ImgRows = str2double(Text(15:end));
            break;
        end
    end
    while ~feof(FID)                                 % 跳过PhoenixHeader
        Text = fgetl(FID);
        if ~isempty(strfind(Text,'[EndofPhoenixHeader]'))
            break
        end
    end
    Mag = fread(FID,ImgColumns*ImgRows,'float32','ieee-be');
    Img = reshape(Mag,[ImgColumns ImgRows]);
    imwrite(uint8(imadjust(Img)*255),[SavePath FileName(1:NameLength-3) 'jpg']); % 调整对比度后保存
    fclose(FID);
end
