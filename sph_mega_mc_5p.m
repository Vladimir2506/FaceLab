% --------------------------------------------------------
% Copyright (c) Weiyang Liu, Yandong Wen
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to evaluate the performance of the trained model on LFW dataset.
% We perform 10-fold cross validation, using cosine similarity as metric.
% More details about the testing protocol can be found at http://vis-www.cs.umass.edu/lfw/#views.
% 
% Usage:
% cd $SPHEREFACE_ROOT/test
% run code/evaluation.m
% --------------------------------------------------------

function evaluation()

clear;clc;close all;
cd('../')

%% caffe setttings
matCaffe = fullfile(pwd, '../tools/caffe-sphereface/matlab');
addpath(genpath(matCaffe));
addpath(genpath('./code/jsonlab-1.5'));
gpu = 1;
if gpu
   gpu_id = 1;
   caffe.set_mode_gpu();
   caffe.set_device(gpu_id);
else
   caffe.set_mode_cpu();
end
caffe.reset_all();

model   = '../train/code/sphereface_deploy.prototxt';
weights = './data/sphereface_premodel.caffemodel';
net     = caffe.Net(model, weights, 'test');
net.save('result/sphereface_model.caffemodel');

%% mtcnn settings
minSize   = 20;
factor    = 0.85;
threshold = [0.6 0.7 0.9];

%% add toolbox paths
pdollarToolbox = fullfile(pwd, '../tools/toolbox');
MTCNN          = fullfile(pwd, '../tools/MTCNN_face_detection_alignment-master/code/codes/MTCNNv1');
addpath(genpath(pdollarToolbox));
addpath(genpath(MTCNN));
modelPath = fullfile(pwd, '../tools/MTCNN_face_detection_alignment-master/code/codes/MTCNNv1/model');
PNet = caffe.Net(fullfile(modelPath, 'det1.prototxt'), fullfile(modelPath, 'det1.caffemodel'), 'test');
RNet = caffe.Net(fullfile(modelPath, 'det2.prototxt'), fullfile(modelPath, 'det2.caffemodel'), 'test');
ONet = caffe.Net(fullfile(modelPath, 'det3.prototxt'), fullfile(modelPath, 'det3.caffemodel'), 'test');

imgSize     = [112, 96];
% coord5point = [30.2946, 51.6963;
%                65.5318, 51.5014;
%                48.0252, 71.7366;
%                33.5493, 92.3655;
%                62.7299, 92.2041];
coord5point = [21.2946, 21.6963;
               74.5318, 21.5014;
               48.0252, 51.7366;
               24.5493, 82.3655;
               71.7299, 82.2041];

%% Distractor features
fprintf('Distractor features.\n');
distractor_file = '/home/4/cm/face_project/datasets/MegaFace/devkit/templatelists/megaface_features_list.json_1000000_1';
d_f = loadjson(distractor_file);
d_f = d_f.path;
time_start = tic;
time_pre = 0;
no_detect_num = 0;
for i = 1:length(d_f)
    if mod(i,10000) == 0
        time_now = toc(time_start);
        time_inter = time_now-time_pre;
        time_pre = time_now;
        fprintf('%d features finished, spending %f s.\n', i, time_inter);
    end
    filename = fullfile('/home/4/cm/face_project/datasets/MegaFace/MegaFace_dataset/FlickrFinal2', cell2mat(d_f(i)));
    name_temp = cell2mat(d_f(i));
    name_temp(end-3:end) = [];
    name_temp = [name_temp, '_sph_5p.mat'];
    featurename = fullfile('/home/4/cm/face_project/datasets/MegaFace/MegaFace_dataset/FlickrFinal2', name_temp);
    
    
    if true % exist(featurename, 'file') == 0 %%%%%%%%%%%%%%%%%%%%%%%%%%%
        no_detect_num = no_detect_num +1;
        img     = imread(filename); 
        if size(img, 3)==1
           img = repmat(img, [1,1,3]);
        end
        % detection
        if i ~= 22108 && i~= 767207 && i ~= 825449
            [bboxes, landmarks] = detect_face(img, minSize, PNet, RNet, ONet, threshold, false, factor);
        else
            facial5point = [];
            bboxes = [];
        end
        if size(bboxes, 1)>1
           % pick the face closed to the center
           center   = size(img) / 2;
           distance = sum(bsxfun(@minus, [mean(bboxes(:, [2, 4]), 2), ...
                                          mean(bboxes(:, [1, 3]), 2)], center(1:2)).^2, 2);
           [~, Ix]  = min(distance);
           facial5point = reshape(landmarks(:, Ix), [5, 2]);
        elseif size(bboxes, 1)==1
           facial5point = reshape(landmarks, [5, 2]);
        else
           facial5point = [];
        end
        if isempty(facial5point)
            continue;
        else
            facial5point = double(facial5point);
            save(featurename, 'facial5point');
        end
    else
        fprintf('The %d th feature_5p exists.\n', i);
    end
end

fprintf('%d features do not have facial5points.\n', no_detect_num);

% %% Probe features
% fprintf('Probe features.\n');
% probe_file = '/home/4/cm/face_project/datasets/MegaFace/devkit/templatelists/facescrub_features_list.json';
% d_f = loadjson(probe_file);
% d_f = d_f.path;
% for i = 1:length(d_f)
%     filename = fullfile('/home/4/cm/face_project/datasets/MegaFace/facescrub_aligned', cell2mat(d_f(i)));
%     name_temp = cell2mat(d_f(i));
%     name_temp(end-3:end) = [];
%     name_temp = [name_temp, '_sph.mat'];
%     featurename = fullfile('/home/4/cm/face_project/datasets/MegaFace/FaceScrubSubset_Features', name_temp);
%     img     = imread(filename);
% 
%     if size(img, 3)==1
%        img = repmat(img, [1,1,3]);
%     end
%     % detection
%     [bboxes, landmarks] = detect_face(img, minSize, PNet, RNet, ONet, threshold, false, factor);
%     if size(bboxes, 1)>1
%        % pick the face closed to the center
%        center   = size(img) / 2;
%        distance = sum(bsxfun(@minus, [mean(bboxes(:, [2, 4]), 2), ...
%                                       mean(bboxes(:, [1, 3]), 2)], center(1:2)).^2, 2);
%        [~, Ix]  = min(distance);
%        facial5point = reshape(landmarks(:, Ix), [5, 2]);
%     elseif size(bboxes, 1)==1
%        facial5point = reshape(landmarks, [5, 2]);
%     else
%        facial5point = [];
%     end
%     if isempty(facial5point)
%        A = [1, 1;        %size(2)first, size(1)then, size(1)>size(2)
%            1, size(img, 1);
%            size(img, 2), 1;
%            size(img, 2), size(img, 1)];
%        B = [1, imgSize(1)-size(img, 1)*imgSize(2)/size(img, 2);
%            1, imgSize(1);
%            imgSize(2), imgSize(1)-ceil(size(img, 1)*imgSize(2)/size(img, 2))
%            imgSize(2), imgSize(1);];
%        transf = cp2tform(A, B, 'similarity');
%        cropImg  = imtransform(img, transf, 'XData', [1 imgSize(2)],...
%                                         'YData', [1 imgSize(1)], 'Size', imgSize);
%     else
%     facial5point = double(facial5point);
%     transf   = cp2tform(facial5point, coord5point, 'similarity');
%     cropImg  = imtransform(img, transf, 'XData', [1 imgSize(2)],...
%                                         'YData', [1 imgSize(1)], 'Size', imgSize);
%     end
%                                     
%     img     = single(cropImg);
% %     if mod(i, 500) == 0
% %         figure;
% %         imshow(uint8(img));
% %     end
%     img     = (img - 127.5)/128;
%     img     = permute(img, [2,1,3]);
%     img     = img(:,:,[3,2,1]);
%     res     = net.forward({img});
%     res_    = net.forward({flip(img, 1)});
%     feature = double([res{1}; res_{1}]);
%     feature = feature/sqrt(sum(feature.^2));
%     save(featurename, 'feature');
% end

% %% Probe features fgnet
% fprintf('Probe features.\n');
% probe_file = '/home/4/cm/face_project/datasets/MegaFace/devkit/templatelists/fgnet_feature_list.json';
% d_f = loadjson(probe_file);
% d_f = d_f.path;
% for i = 1:length(d_f)
%     filename = fullfile('/home/4/cm/face_project/datasets/MegaFace/FGNET2', cell2mat(d_f(i)));
%     name_temp = cell2mat(d_f(i));
%     name_temp(end-3:end) = [];
%     name_temp = [name_temp, '_sph.mat'];
%     featurename = fullfile('/home/4/cm/face_project/datasets/MegaFace/FGNET2', name_temp);
%     img     = imread(filename);
% 
%     if size(img, 3)==1
%        img = repmat(img, [1,1,3]);
%     end
%     % detection
%     [bboxes, landmarks] = detect_face(img, minSize, PNet, RNet, ONet, threshold, false, factor);
%     if size(bboxes, 1)>1
%        % pick the face closed to the center
%        center   = size(img) / 2;
%        distance = sum(bsxfun(@minus, [mean(bboxes(:, [2, 4]), 2), ...
%                                       mean(bboxes(:, [1, 3]), 2)], center(1:2)).^2, 2);
%        [~, Ix]  = min(distance);
%        facial5point = reshape(landmarks(:, Ix), [5, 2]);
%     elseif size(bboxes, 1)==1
%        facial5point = reshape(landmarks, [5, 2]);
%     else
%        facial5point = [];
%     end
%     if isempty(facial5point)
%        A = [1, 1;        %size(2)first, size(1)then, size(1)>size(2)
%            1, size(img, 1);
%            size(img, 2), 1;
%            size(img, 2), size(img, 1)];
%        B = [1, imgSize(1)-size(img, 1)*imgSize(2)/size(img, 2);
%            1, imgSize(1);
%            imgSize(2), imgSize(1)-ceil(size(img, 1)*imgSize(2)/size(img, 2))
%            imgSize(2), imgSize(1);];
%        transf = cp2tform(A, B, 'similarity');
%        cropImg  = imtransform(img, transf, 'XData', [1 imgSize(2)],...
%                                         'YData', [1 imgSize(1)], 'Size', imgSize);
%     else
%     facial5point = double(facial5point);
%     transf   = cp2tform(facial5point, coord5point, 'similarity');
%     cropImg  = imtransform(img, transf, 'XData', [1 imgSize(2)],...
%                                         'YData', [1 imgSize(1)], 'Size', imgSize);
%     end
%                                     
%     img     = single(cropImg);
% %     if mod(i, 500) == 0
% %         figure;
% %         imshow(uint8(img));
% %     end
%     img     = (img - 127.5)/128;
%     img     = permute(img, [2,1,3]);
%     img     = img(:,:,[3,2,1]);
%     res     = net.forward({img});
%     res_    = net.forward({flip(img, 1)});
%     feature = double([res{1}; res_{1}]);
%     feature = feature/sqrt(sum(feature.^2));
%     save(featurename, 'feature');
% end