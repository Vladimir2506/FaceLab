% Mark all landmarks down
% Special for IMDB 
function GetLandmark()

    % Clean up environment
    clear;
    clc;
    close all;

    % Setup environment
    cd('./');
    gpu = 1;
    gpuID = 1;
    pathMTCNN = '/home/xjch/Desktop/MTCNN_face_detection_alignment-master/code/codes/MTCNNv2';
    pathCaffe = '/home/xjch/Desktop/caffe/matlab';
    pathToolbox = '/home/xjch/Desktop/toolbox-master';
    pathModels = '/home/xjch/Desktop/MTCNN_face_detection_alignment-master/code/codes/MTCNNv2/model';
    addpath(genpath(pathCaffe));
    addpath(genpath(pathToolbox));
    addpath(genpath(pathMTCNN));

    %Load data from metadata:
    %imdb.mat
    load('imdb.mat');
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
    imgZeroFace = {};
    imgMultiFace = {};
    
    for i = 1 : length(imgList)
	    
        tic;

        % Execute MTCNN FP
        img = imread(imgList{i});
		if(numel(size(img)) == 2)
			img = cat(3, img, img, img);
		end
        [imgPath, imgName, ~] = fileparts(imgList{i});
        lmName = [ imgPath, '/', imgName, '_fp.mat'];
		bbName = [ imgPath, '/', imgName, '_bb.mat'];
        [BBoxes, landmarks] = ...
            detect_face(img, minsize, PNet, RNet, ONet, LNet, ...
                threshold, false, factor);
        imgProceeded = imgProceeded + 1;
        
        % Deal with boundingboxes
        detected = size(BBoxes, 1);
        faceDetected = faceDetected + detected;
        
        % Deal with landmarks
        if detected == 0
            imgZeroFace{end + 1} = imgList{i};
        else
            if detected > 1
                largest = 1;
                maxArea = (BBoxes(1,3) - BBoxes(1,1)) * (BBoxes(1,4) - BBoxes(1,2));
                for k = 2 : detected
                    thisArea = (BBoxes(k,3) - BBoxes(k,1)) * (BBoxes(k,4) - BBoxes(k,2));
                    if thisArea > maxArea
                        maxArea = thisArea;
                        largest = k;
                    end
                end
               
                facial5point = double(reshape(landmarks(:, largest), [5, 2]));
				boundingbox = double(reshape(BBoxes(largest, 1:4), [1, 4]));
                save(lmName, 'facial5point');
                save(bbName, 'boundingbox');
                imgMultiFace{end + 1} = imgList{i};
            else
                facial5point = double(reshape(landmarks, [5, 2]));
				boundingbox = double(reshape(BBoxes(1:4), [1, 4]));
                save(lmName, 'facial5point'); 
                save(bbName, 'boundingbox');
            end
    
        end
        
        fprintf('%d th image proceeded.\n', imgProceeded);
        toc;
    end

    %Report
    fprintf('%d images are proceeded, %d faces are detected.\n', imgProceeded, faceDetected);
    
    if(~isempty(imgZeroFace))
        zFaces = size(imgZeroFace, 1);
        fprintf('%d images are detected no face.\n', zFaces);
        for j = 1 : zFaces
            fprintf('%s\n', cell2str(imgZeroFace{j}));
            save('zeroFaces', imgZeroFace);
        end
    end
    
    if(~isempty(imgMultiFace))
        mFaces = size(imgMultiFace, 1);
        fprintf('%d images are detected more than one faces.\n', mFaces);
        for j = 1 : mFaces
            fprintf('%s\n', cell2str(imgMultiFace{j}));
            save('multiFaces', imgMultiFace);
        end
    end

end

