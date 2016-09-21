function [ x,y ] = computeInterPoint( line1,line2 )
%计算线1和线2的交叉点，其中线一是直线，线2是斜线
%   Detailed explanation goes here
y=line1(1);
x=(y-line2(2))/line2(1);%即x=(y-b)/k;
end

