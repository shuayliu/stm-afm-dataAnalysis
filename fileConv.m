function filename = fileConv(path,filename)
%去除文本文件中的文字行，只保留二、三列数据，并将Distance列（第二列）数据×1E9
%返回值为新的filename

fidin = fopen(strcat(path,filename));
tmp=[];
while ~feof(fidin)
    tline=fgetl(fidin);
    if double(tline(1)) >= 48 && double(tline(1)) <= 57
        tmp=[tmp;str2num(tline)];
        continue
    end
end
fclose(fidin);
%只保留二、三列数据
[row,col]=size(tmp);
if col>2 
    tmp=tmp(:,2:3);
%Distance列数据×1E9
tmp(:,1)=tmp(:,1)*1E9;

%重命名文件为原文件名conv.txt
filename=strcat(strcat('Conv_',strtok(filename,'.')),'.txt');
%保存文件
save(strcat(path,filename),'tmp','-ascii');
%清除内存
clear tmp;
end

