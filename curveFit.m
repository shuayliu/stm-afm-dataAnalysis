function [ line1,line2,pos ] = curveFit( Data )
%用两条直线拟合数据集Data,返回拐点的索引用于绘图
[row,col]=size(Data);
data_x=Data(1:row,1);
data_y=Data(1:row,2);

%   首先对data_x,data_y进行降维处理,这里采用均匀采样法，每隔10个点采样一次
sample_x = zeros(row/10,1);
sample_y = zeros(row/10,1);

  for i=1:1:row/10
      sample_x(i)=data_x(i*10);
      sample_y(i)=data_y(i*10);
  end
% 
% 
%下面需要找到最大的拐点，将数据分为两部分，分别进行曲线拟合
pos=maxDistance([sample_x,sample_y]);
pos=pos*10 ;
% 
% part1_x=data_x(1:pos);
% part1_y=data_y(1:pos);
% 
% part2_x=data_x(pos+1:row);
% part2_y=data_y(pos+1:row);

% 在4000点数据中，
% 用第51个点到第100个点的数据拟合水平直线，

for i=1:1:50
    part1_x(i)=data_x(i+50);
    part1_y(i)=data_y(i+50);
end
% % 用第1901个点到第1950个点的数据拟合另一条直线，
% % 其他处理和原来1000点、2000点数据的处理方式相同。
% for i=1:1:50
%     part2_x(i)=data_x(i+1900);
%     part2_y(i)=data_y(i+1900);
% end

% 用第row-100个点到第row-50个点的数据拟合另一条直线，
% 其他处理和原来1000点、2000点数据的处理方式相同。
for i=1:1:50
    part2_x(i)=data_x(i+row-100);
    part2_y(i)=data_y(i+row-100);
end


%下面对这两部分数据进行拟合
line1=polyfit(part1_x,part1_y,0);%直线
line2=polyfit(part2_x,part2_y,1);%斜线



end

