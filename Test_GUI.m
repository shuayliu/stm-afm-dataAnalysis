function varargout = Test_GUI(varargin)
% TEST_GUI MATLAB code for Test_GUI.fig
%      TEST_GUI, by itself, creates a new TEST_GUI or raises the existing
%      singleton*.
%
%      H = TEST_GUI returns the handle to a new TEST_GUI or the handle to
%      the existing singleton*.
%
%      TEST_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TEST_GUI.M with the given input arguments.
%
%      TEST_GUI('Property','Value',...) creates a new TEST_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Test_GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Test_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Test_GUI

% Last Modified by GUIDE v2.5 19-Sep-2016 13:55:18

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Test_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @Test_GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Test_GUI is made visible.
function Test_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Test_GUI (see VARARGIN)

% Choose default command line output for Test_GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Test_GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Test_GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in loadButton.
function loadButton_Callback(hObject, eventdata, handles)
% hObject    handle to loadButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global singleFile ;
   global filename;
   global dataNum;
   global pathname;
   %---------------------remeber open pathname---------
   if ~ischar(pathname)
       pathname=cd();
   end
   [filename,pathname]=uigetfile(strcat(pathname,'\*.txt'));
   open_pathname=pathname;
   %----------------------modify by Jonahliu---------
   %-------------------------------------
   if isequal(filename,0) || isequal(open_pathname,0)
   else
      %----------change origin file format----------
      filename=fileConv(open_pathname,filename);
      %-----------add by JonahLiu@stm ---------------
      file_name=strcat(open_pathname,filename);
      singleFile=importdata(file_name) ;
      dataNum = length(singleFile(:,1));
      %plot the data of singleFile in the axes_origin
      x1=singleFile(1:dataNum/2,1);
      y1=singleFile(1:dataNum/2,2);

      x2=singleFile(dataNum/2+1:dataNum,1);
      y2=singleFile(dataNum/2+1:dataNum,2);

      axes(handles.axes_origin);
      cla reset;
      plot(x1,y1,'r.');
      hold on;
      plot(x2,y2,'b.');
   end


% --- Executes on button press in closeButton.
function closeButton_Callback(hObject, eventdata, handles)
% hObject    handle to closeButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close(handles.figure1);



function slopeEdit_Callback(hObject, eventdata, handles)
% hObject    handle to slopeEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of slopeEdit as text
%        str2double(get(hObject,'String')) returns contents of slopeEdit as a double


% --- Executes during object creation, after setting all properties.
function slopeEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slopeEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function XEdit_Callback(hObject, eventdata, handles)
% hObject    handle to XEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of XEdit as text
%        str2double(get(hObject,'String')) returns contents of XEdit as a double


% --- Executes during object creation, after setting all properties.
function XEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to XEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function YEdit_Callback(hObject, eventdata, handles)
% hObject    handle to YEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of YEdit as text
%        str2double(get(hObject,'String')) returns contents of YEdit as a double


% --- Executes during object creation, after setting all properties.
function YEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to YEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function varEdit_Callback(hObject, eventdata, handles)
% hObject    handle to varEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of varEdit as text
%        str2double(get(hObject,'String')) returns contents of varEdit as a double


% --- Executes during object creation, after setting all properties.
function varEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to varEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in mutilFileButton.
function mutilFileButton_Callback(hObject, eventdata, handles)
% hObject    handle to mutilFileButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
      path=uigetdir;
      filelist=dir(fullfile(path,'*.txt'));
      path=strcat(path,'\');
      n=length(filelist);
      if n==0
          msgbox('can not find the file!');
      else
          resultPath=uigetdir(); 
          if ~isequal(resultPath,0)
             h=waitbar(0,'pleas wait...');
            for i=1:n
                %-------------modify by Jonahliu@stm----------
                new_filename=fileConv(path,filelist(i).name);
                %-----------------------------------------
                filename=strcat(path,new_filename);

                data = load(filename);
                dataNum=length(data(:,1));
                [line1, line2,pos]=curveFit([data(1:dataNum/2,1),data(1:dataNum/2,2)]);
                var = str2double(get(handles.varEdit,'String'));
                [x,y]=computeInterPoint(line1,line2);
                [X,Y]=covertData(data(1:dataNum/2,1:2),line2(1),x,y,var);
                resultFileName=strcat(resultPath,'\');
                resultFileName=strcat( resultFileName,filelist(i).name);
                fid=fopen(resultFileName,'w+');
                for j=1:1:dataNum/2
                    fprintf(fid,'%f   %f\n',X(j),Y(j));
                end
                fclose(fid); 
                waitbar(i/n,h);
            end    
            close(h);
            msgbox('ok!');
          end
      end



function numXEdit_Callback(hObject, eventdata, handles)
% hObject    handle to numXEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of numXEdit as text
%        str2double(get(hObject,'String')) returns contents of numXEdit as a double


% --- Executes during object creation, after setting all properties.
function numXEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to numXEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in histogramButton.
function histogramButton_Callback(hObject, eventdata, handles)
% hObject    handle to histogramButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global dataHistogram;
      global histogramN;
      path=uigetdir;
      dataHistogram=[];
      if ~isequal(path,0)
        filelist=dir(fullfile(path,'*.txt'));
        path=strcat(path,'\');
        n=length(filelist);
        if n==0
            msgbox('read fail!');
        else
             for i=1:n
                %-------------modify by Jonahliu@stm----------
                new_filename=fileConv(path,filelist(i).name);
                %-----------------------------------------
                filename=strcat(path,new_filename);
                temp = load(filename);
                dataHistogram=[dataHistogram;temp];
             end    
        end
      end
      %rank the dataHistogram by ascend according to the x coordinateимим 
      [row col]=size(dataHistogram);
      X=dataHistogram(:,1);
      Y=dataHistogram(:,2);
      [new_X,index]=sort(X,'ascend');
      new_Y=Y;

      for i=1:1:row
         new_Y(i)=Y(index(i));
      end
      dataHistogram=[new_X,new_Y];
      plot(dataHistogram(:,1),dataHistogram(:,2),'.');
      
      %plots a histogram using an numGrid-by-numYGrid grid of bins
      numXGrid=str2double(get(handles.numXEdit,'String'));
      numYGrid=str2double(get(handles.numYEdit,'String'));
      
      xMin=min(dataHistogram(:,1));
      xMax=max(dataHistogram(:,1));
      cell=(xMax-xMin)/numXGrid;
      
     figure;
     histogramN= hist3([dataHistogram(:,1),dataHistogram(:,2)],[numXGrid numYGrid]);
     hist3([dataHistogram(:,1),dataHistogram(:,2)]);
     xlabel('x');
     ylabel('y');
     zlabel('z');
     figure;
     newN=histogramN';
    
     bar3c(newN,1);
     xlabel('x');
     ylabel('y');
     zlabel('z');
     view(0,-90);
     colorbar;


% --- Executes on button press in updateButton.
function updateButton_Callback(hObject, eventdata, handles)
% hObject    handle to updateButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global dataHistogram;
     global histogramN;
     %get the minXNum,maxXNum,numGrid,and the same of Y
     minXNum=str2double(get(handles.minXEdit,'String'));
     maxXNum=str2double(get(handles.maxXEdit,'String'));
     numXGrid=str2double(get(handles.numXEdit,'String'));
     
     minYNum=str2double(get(handles.minYEdit,'String'));
     maxYNum=str2double(get(handles.maxYEdit,'String'));
     numYGrid=str2double(get(handles.numYEdit,'String'));
     
     [row col]=size(dataHistogram);
     dataNum=row;
     X=dataHistogram(:,1);
     
     pos_min=1;
     pos_max=row;
     %restrict xaccording to the minXNum and maxXNum
    if minXNum~=maxXNum
         for i=1:1:row-1
            if X(i)<=minXNum && X(i+1)>minXNum
                pos_min=i;
            end
            if X(row-i+1)>maxXNum && X(row-i)<maxXNum
                pos_max=row-i;
            end
         end
    end
    X=dataHistogram(pos_min:pos_max,1);
    Y=dataHistogram(pos_min:pos_max,2);
    
    %restrict yaccording to the minYNum and maxYNum, we exchange the X and Y in order to reuse the code above,then we only  need to restrict the X according to the minYNum and maxYNum
    if minYNum~=maxYNum
        temp=[Y,X];
        [row col]=size(temp);
         X=temp(:,1);
         Y=temp(:,2);
         [new_X,index]=sort(X,'ascend');
         new_Y=Y;
        for i=1:1:row
         new_Y(i)=Y(index(i));
        end
        temp=[new_X,new_Y];
          X=temp(:,1);
          Y=temp(:,2);  
        %after rank the data we need to restrict it by minYNumandmaxYNum
        pos_min=1;
        pos_max=row;
        for i=1:1:row-1
            if X(i)<=minYNum && X(i+1)>minYNum
                pos_min=i;
            end
            if X(row-i+1)>maxYNum && X(row-i)<maxYNum
                pos_max=row-i;
            end
        end
        X=X(pos_min:pos_max);
        Y=Y(pos_min:pos_max);

        %exchange X,Y
        temp2=X;
        X=Y;
        Y=temp2;
    end
    
    %to avoid the min of X/Y can not reach minXEdit/minYEdit
    if pos_min==1
        X(1)=minXNum;
        Y(1)=minYNum;
    end
    %to avoid the max of X/Y can not reach maxXEdit/maxYEdit
    if pos_max==row
        X(row)=manXNum;
        Y(row)=manYNum;
    end
    
    
     %show result with plot , hist3 and bar3c
     cla reset;
     plot(X,Y,'.');
    
     figure;
     if dataNum~=0
         histogramN= hist3([X,Y],[numXGrid numYGrid]);
         hist3([X,Y],[numXGrid numYGrid]);
        xlabel('x');
        ylabel('y');
        zlabel('z');
    
        figure;
        newN=histogramN';
         bar3c(newN,1);
         xlabel('x');
         ylabel('y');
         zlabel('z');
         view(0,-90);
         colorbar;
     end
     



function minXEdit_Callback(hObject, eventdata, handles)
% hObject    handle to minXEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of minXEdit as text
%        str2double(get(hObject,'String')) returns contents of minXEdit as a double


% --- Executes during object creation, after setting all properties.
function minXEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to minXEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function maxXEdit_Callback(hObject, eventdata, handles)
% hObject    handle to maxXEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of maxXEdit as text
%        str2double(get(hObject,'String')) returns contents of maxXEdit as a double


% --- Executes during object creation, after setting all properties.
function maxXEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to maxXEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function numYEdit_Callback(hObject, eventdata, handles)
% hObject    handle to numYEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of numYEdit as text
%        str2double(get(hObject,'String')) returns contents of numYEdit as a double


% --- Executes during object creation, after setting all properties.
function numYEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to numYEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function minYEdit_Callback(hObject, eventdata, handles)
% hObject    handle to minYEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of minYEdit as text
%        str2double(get(hObject,'String')) returns contents of minYEdit as a double


% --- Executes during object creation, after setting all properties.
function minYEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to minYEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function maxYEdit_Callback(hObject, eventdata, handles)
% hObject    handle to maxYEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of maxYEdit as text
%        str2double(get(hObject,'String')) returns contents of maxYEdit as a double


% --- Executes during object creation, after setting all properties.
function maxYEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to maxYEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in storeHistogramButton.
function storeHistogramButton_Callback(hObject, eventdata, handles)
% hObject    handle to storeHistogramButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 global histogramN;
        [filename,path]=uiputfile();
        filename=strcat(path,filename);
        fid=fopen(filename,'w+');   
        [row col]=size(histogramN);
        for i=1:1:col
            for j=1:1:row
                fprintf(fid,'%f\n',histogramN(j,i));
            end
        end
        fclose(fid);


function minEdit_Callback(hObject, eventdata, handles)
% hObject    handle to minEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of minEdit as text
%        str2double(get(hObject,'String')) returns contents of minEdit as a double


% --- Executes during object creation, after setting all properties.
function minEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to minEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function maxEdit_Callback(hObject, eventdata, handles)
% hObject    handle to maxEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of maxEdit as text
%        str2double(get(hObject,'String')) returns contents of maxEdit as a double


% --- Executes during object creation, after setting all properties.
function maxEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to maxEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in fitButton.
function fitButton_Callback(hObject, eventdata, handles)
% hObject    handle to fitButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global singleFile ;
global dataNum;
[line1, line2,pos]=curveFit([singleFile(1:dataNum/2,1),singleFile(1:dataNum/2,2)])
    %draw the origin data
  x1=singleFile(1:dataNum/2,1);
  y1=singleFile(1:dataNum/2,2);

  x2=singleFile(dataNum/2+1:dataNum,1);
  y2=singleFile(dataNum/2+1:dataNum,2);

  axes(handles.axes_origin);
  cla reset;
  plot(x1,y1,'r.');
  hold on;
  plot(x2,y2,'b.');
%
    %draw the fitting curve
    x=singleFile(1:pos+50,1);
    y=polyval(line1,x);
    hold on;
    plot(x,y,'k');
    hold on;
    x=singleFile(pos-50:dataNum/2,1);
    y=polyval(line2,x);
    plot(x,y,'k');


% --- Executes on button press in convertButton.
function convertButton_Callback(hObject, eventdata, handles)
% hObject    handle to convertButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global singleFile;
global dataNum;
global X;
global Y;

data=singleFile(1:dataNum/2,1:2);
[line1, line2,pos]=curveFit([singleFile(1:dataNum/2,1),singleFile(1:dataNum/2,2)]);
var = str2double(get(handles.varEdit,'String'));
[x,y]=computeInterPoint(line1,line2);
[X,Y]=covertData(data,line2(1),x,y,var);
axes(handles.axes_origin);

minNum=str2double(get(handles.minEdit,'String'));
maxNum=str2double(get(handles.maxEdit,'String'));

%restrict the range of x by minNum and maxNum
pos_min=1;
pos_max=dataNum/2;
if minNum~=maxNum
    for i=1:1:dataNum/2-1
        if X(i)<=minNum && X(i+1)>minNum
            pos_min=i;
        end
        if X(dataNum/2-i+1)>maxNum && X(dataNum/2-i)<maxNum
            pos_max=dataNum/2-i;
        end
    end
end
X=X(pos_min:pos_max);
Y=Y(pos_min:pos_max);
cla reset;
plot(X,Y,'b.');


% --- Executes on button press in pointButton.
function pointButton_Callback(hObject, eventdata, handles)
% hObject    handle to pointButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global singleFile ;
global dataNum;
[line1, line2,pos]=curveFit([singleFile(1:dataNum/2,1),singleFile(1:dataNum/2,2)])

[x,y]=computeInterPoint(line1,line2);
set(handles.slopeEdit,'String',num2str(line2(1)));
set(handles.XEdit,'String',num2str(x));
set(handles.YEdit,'String',num2str(y));


% --- Executes on button press in storeButton.
function storeButton_Callback(hObject, eventdata, handles)
% hObject    handle to storeButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
      global filename;
      global X;
      global Y;
      global store_pathname;
      global pathname;
      %------------modify be Jonahliu@stm------
      %------------remember last store path---
      if ~ischar(store_pathname)
          store_pathname=pathname;
      end
      if ~isequal(filename,0)       
      [filename,store_pathname]=uiputfile(strcat(strcat(store_pathname,'\'),filename) );
       if isequal(filename,0) | isequal(store_pathname,0)
            msgbox('User pressed cancel');
       else
            resultFileName=strcat(store_pathname,strcat('Trans_',filename));
            fid=fopen(resultFileName,'w+');
            for i=1:1:length(X)
                 fprintf(fid,'%f %f\n',X(i),Y(i));
            end
             fclose(fid);
            msgbox('ok!');
       end
      end
   



% --- Executes on button press in lookButton.
function lookButton_Callback(hObject, eventdata, handles)
% hObject    handle to lookButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename,pathname]=uigetfile('*.txt');
    if isequal(filename,0) || isequal(pathname,0)
    else
            %-----------------modify by JonahLiu@stm---------
            file_name=strcat(pathname,fileConv('',filename));
            %----------------------
            data=load(file_name) ;
            cla reset;
            plot(data(:,1),data(:,2),'.');
     end
