% Mark all landmarks down
% Special for IMDB 
function MarkLandmarks()

    % Clean up environment
    clear;
    clc;
    close all;

    % Setup environment
    gpu = 1;
    gpuID = 7;
    pathMTCNN = '/home/xjch/Desktop/MTCNN_face_detection_alignment-master/code/codes/MTCNNv2';
    pathCaffe = '/home/xjch/Desktop/caffe/matlab';
    pathToolbox = '/home/xjch/Desktop/toolbox-master';
    pathModels = '/home/xjch/Desktop/MTCNN_face_detection_alignment-master/code/codes/MTCNNv2/mdoel';
    addpath(genpath(pathCaffe));
    addpath(genpath(pathToolbox));
    addpath(genpath(pathMTCNN));

    %Load data from metadata:
    %imdb.mat
    load(imdb.mat);
    imgList = imdb.full_path;

    % Setup MTCNN
    threshold = [0.6 0.7 0.9];
    factor = 0.85;
    minsize = 20;

    % Startup caffe model
    if gpu
        caffe.set_mode_gpu();
        caffe.set_device(gpuID);
    else
        caffe.set_mode_cpu();
    end

    caffe.reset_all();

    prototxt_dir = strcat(pathModels,'/det1.prototxt');
    model_dir = strcat(pathModels,'/det1.caffemodel'); 
    PNet = caffe.Net(prototxt_dir,model_dir,'test');

    prototxt_dir = strcat(pathModels,'/det2.prototxt');
    model_dir = strcat(pathModels,'/det2.caffemodel');
    RNet = caffe.Net(prototxt_dir,model_dir,'test');	

    prototxt_dir = strcat(pathModels,'/det3.prototxt');
    model_dir = strcat(pathModels,'/det3.caffemodel');
    ONet = caffe.Net(prototxt_dir,model_dir,'test');

    prototxt_dir =  strcat(pathModels,'/det4.prototxt');
    model_dir =  strcat(pathModels,'/det4.caffemodel');
    LNet = caffe.Net(prototxt_dir,model_dir,'test');
	
    % Run MTCNN and get landmarks
    imgProceeded = 0;
    faceDetected = 0;
    imgZeroFace = [];
    imgMultiFace = [];
    
    for i = 1 : 2%length(imgList)
	    
        img = imread(imgList{i});
        [imgPath, imgName, imgExt, imgVer] = fileparts(imgList{i});
        lmName = [imgName, '_sph_5p.mat'];
        [BBoxes, landmarks] = 
            detect_face(img, minsize, PNet, RNet, ONet, LNet, 
                threshold, false, factor);
        imgProceeded += 1;

        % Deal with boundingboxes
        detected = size(boundingboxes, 1);
        faceDetected += detected;
        
        % Deal with landmarks
        if detected == 0
            imgZeroFace = [imgZeroFace; imgList{i}];
        else if detected > 1
            imgMultiFace = [imgMultiFace; imgList{i}];
        else
            facial5 = reshape(landmarks, [5, 2]);
            facial5 = double(facial5);
            save(lmName, 'facial5');
        end
    
    end

    %Report
    fprintf('%d images are proceeded, %d faces are detected.\n', imgProceeded, faceDetected);
    
    if(~isempty(imgZeroFace))
        zFaces = size(imgZeroFace, 1);
        fprintf('%d images are detected no face.\n', zFaces);
        for j = 1 : zFaces
            fprintf('%s\n', imgZeroFace{j});
        end
    end
    
    if(~isempty(imgMultiFace))
        mFaces = size(imgMultiFace, 1);
        fprintf('%d images are detected more than one faces.\n', mFaces);
        for j = 1 : mFaces
            fprintf('%s\n', imgMultiFace{j});
        end
    end

end