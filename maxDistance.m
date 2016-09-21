function pos = maxDistance(Data)
%寻找到由两个端点确定的直线的距离最大的点,返回该点的索引

x=[];
y=[];

[row,col]=size(Data);

%获取两个端点(x1,y1),(x2,y2)
x(1)=Data(1,1);
x(2)=Data(row,1);
y(1)=Data(1,2);
y(2)=Data(row,2);

p=polyfit(x,y,1);%得到由两点确定的直线,p(1)是斜率，p(2)是常数
%计算到直线距离最大的点
pos=1;
max=0;
for i=1:1:row
    distance = abs(p(1)*Data(i,1)-Data(i,2)+p(2))/sqrt(p(1)*p(1)+1) ;
    if distance>max
        max=distance;
        pos=i;
    end
end
end

