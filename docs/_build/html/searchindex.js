Search.setIndex({docnames:["RAVE","RAVE.common","RAVE.eye_tracker","RAVE.facial_detection","examples","index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["RAVE.rst","RAVE.common.rst","RAVE.eye_tracker.rst","RAVE.facial_detection.rst","examples.rst","index.rst"],objects:{"RAVE.common":[[1,0,0,"-","Dataset"],[1,0,0,"-","DatasetBuilder"],[1,0,0,"-","Trainer"],[1,0,0,"-","image_utils"]],"RAVE.common.Dataset":[[1,1,1,"","Dataset"]],"RAVE.common.Dataset.Dataset":[[1,2,1,"","DATASET_DIR"],[1,2,1,"","IMAGES_DIR"],[1,2,1,"","IMAGES_FILE_EXTENSION"],[1,2,1,"","LABELS_DIR"],[1,2,1,"","TEST_DIR"],[1,2,1,"","TRAINING_DIR"],[1,2,1,"","VALIDATION_DIR"],[1,3,1,"","__getitem__"],[1,3,1,"","__init__"],[1,3,1,"","__len__"],[1,2,1,"","__module__"],[1,2,1,"","__parameters__"],[1,3,1,"","get_image_and_label_on_disk"],[1,3,1,"","get_multiple_workers_safe_list_of_paths"],[1,3,1,"","get_test_sub_dataset"],[1,3,1,"","get_training_sub_dataset"],[1,3,1,"","get_validation_sub_dataset"]],"RAVE.common.DatasetBuilder":[[1,1,1,"","DatasetBuilder"]],"RAVE.common.DatasetBuilder.DatasetBuilder":[[1,2,1,"","ANNOTATIONS_DIR"],[1,2,1,"","ROOT_PATH"],[1,2,1,"","VIDEOS_DIR"],[1,3,1,"","create_directory_if_does_not_exist"],[1,3,1,"","create_images_dataset_with_one_video"],[1,3,1,"","create_images_of_one_video_group"],[1,3,1,"","get_builders"],[1,3,1,"","process_frame"],[1,3,1,"","process_image_label_pair"],[1,3,1,"","save_image_label_pair"]],"RAVE.common.Trainer":[[1,1,1,"","Trainer"]],"RAVE.common.Trainer.Trainer":[[1,2,1,"","MODEL_INFO_FILE_NAME"],[1,2,1,"","TRAINING_SESSIONS_DIR"],[1,3,1,"","compute_training_loss"],[1,3,1,"","compute_validation_loss"],[1,3,1,"","load_best_model"],[1,3,1,"","load_model_and_training_info"],[1,3,1,"","save_model_and_training_info"],[1,3,1,"","terminate_training_thread"],[1,3,1,"","train_with_validation"],[1,3,1,"","update_plot"]],"RAVE.common.image_utils":[[1,4,1,"","apply_image_rotation"],[1,4,1,"","apply_image_translation"],[1,4,1,"","apply_image_translation_and_rotation"],[1,4,1,"","box_iou"],[1,4,1,"","clip_coords"],[1,4,1,"","do_affine_grid_operation"],[1,4,1,"","intersection"],[1,4,1,"","inverse_normalize"],[1,4,1,"","opencv_image_to_tensor"],[1,4,1,"","scale_coords"],[1,4,1,"","scale_coords_landmarks"],[1,4,1,"","tensor_to_opencv_image"],[1,4,1,"","xywh2xyxy"],[1,4,1,"","xyxy2xywh"]],"RAVE.eye_tracker":[[2,0,0,"-","EyeTrackerDataset"],[2,0,0,"-","EyeTrackerDatasetBuilder"],[2,0,0,"-","EyeTrackerModel"],[2,0,0,"-","NormalizedEllipse"],[2,0,0,"-","ellipse_util"]],"RAVE.eye_tracker.EyeTrackerDataset":[[2,1,1,"","EyeTrackerDataset"],[2,1,1,"","EyeTrackerDatasetOnlineDataAugmentation"]],"RAVE.eye_tracker.EyeTrackerDataset.EyeTrackerDataset":[[2,2,1,"","EYE_TRACKER_DIR_PATH"],[2,2,1,"","IMAGE_DIMENSIONS"],[2,2,1,"","TRAINING_MEAN"],[2,2,1,"","TRAINING_STD"],[2,3,1,"","get_test_sub_dataset"],[2,3,1,"","get_training_sub_dataset"],[2,3,1,"","get_validation_sub_dataset"]],"RAVE.eye_tracker.EyeTrackerDatasetBuilder":[[2,1,1,"","EyeTrackerDatasetBuilder"],[2,1,1,"","EyeTrackerDatasetBuilderOfflineDataAugmentation"]],"RAVE.eye_tracker.EyeTrackerDatasetBuilder.EyeTrackerDatasetBuilder":[[2,3,1,"","create_images_datasets_with_LPW_videos"],[2,3,1,"","get_builders"],[2,3,1,"","parse_current_annotation"],[2,3,1,"","process_image_label_pair"]],"RAVE.eye_tracker.EyeTrackerDatasetBuilder.EyeTrackerDatasetBuilderOfflineDataAugmentation":[[2,3,1,"","apply_translation_and_rotation"],[2,3,1,"","process_frame"]],"RAVE.eye_tracker.EyeTrackerModel":[[2,1,1,"","EyeTrackerModel"]],"RAVE.eye_tracker.EyeTrackerModel.EyeTrackerModel":[[2,3,1,"","forward"],[2,2,1,"","training"]],"RAVE.eye_tracker.NormalizedEllipse":[[2,1,1,"","NormalizedEllipse"]],"RAVE.eye_tracker.NormalizedEllipse.NormalizedEllipse":[[2,3,1,"","get_from_list"],[2,3,1,"","get_from_opencv_ellipse"],[2,3,1,"","rotate_around_image_center"],[2,3,1,"","to_list"]],"RAVE.eye_tracker.ellipse_util":[[2,4,1,"","draw_ellipse_on_image"],[2,4,1,"","ellipse_loss_function"],[2,4,1,"","get_points_of_an_ellipse"],[2,4,1,"","get_points_of_ellipses"]],"RAVE.face_detection":[[3,0,0,"-","FaceDetectionDataset"],[3,0,0,"-","FaceDetectionModel"],[3,0,0,"-","fpsHelper"]],"RAVE.face_detection.FaceDetectionDataset":[[3,1,1,"","FaceDetectionDataset"]],"RAVE.face_detection.FaceDetectionDataset.FaceDetectionDataset":[[3,2,1,"","FACE_DETECTION_DIR_PATH"],[3,2,1,"","IMAGE_DIMENSIONS"],[3,2,1,"","TRAINING_MEAN"],[3,2,1,"","TRAINING_STD"],[3,3,1,"","get_test_sub_dataset"],[3,3,1,"","get_training_sub_dataset"],[3,3,1,"","get_validation_sub_dataset"]],"RAVE.face_detection.FaceDetectionModel":[[3,1,1,"","FaceDetectionModel"]],"RAVE.face_detection.FaceDetectionModel.FaceDetectionModel":[[3,2,1,"","CONFIDENCE_THRESHOLD"],[3,2,1,"","INTERSECTION_OVER_UNION_THRESHOLD"],[3,3,1,"","forward"],[3,2,1,"","training"]],"RAVE.face_detection.fpsHelper":[[3,1,1,"","FPS"]],"RAVE.face_detection.fpsHelper.FPS":[[3,3,1,"","getFps"],[3,3,1,"","setFps"],[3,3,1,"","start"],[3,3,1,"","writeFpsToFrame"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:function"},terms:{"0":[1,2,3],"0119":3,"0581":3,"0609":3,"0648":3,"0695":3,"095":3,"1":[1,2,3],"2":[1,2],"224":2,"225":2,"229":2,"240":2,"255":2,"2645689":2,"2850f3cd465155689052f0fa3a177a50":1,"299":2,"3":[1,2,3],"4":1,"406":2,"456":2,"480":2,"485":2,"5":[2,3],"800":3,"abstract":1,"class":[1,2,3],"default":[1,2],"do":2,"float":[1,2],"function":[1,2],"int":[1,2],"new":2,"return":[1,2,3],"static":[1,2,3],A:[1,2],For:2,It:[1,2],One:1,The:[1,2,3,4],To:[1,2,3],__getitem__:1,__init__:1,__len__:1,__module__:1,__parameters__:1,_lrschedul:1,abc:1,accessor:3,affin:1,after:3,aid:5,aim:5,all:[1,2],alongsid:1,alreadi:2,also:[1,2],amplitud:1,an:[1,2,3],angl:[1,2],annot:[1,2],annotations_dir:1,appli:[1,2],apply_image_rot:1,apply_image_transl:1,apply_image_translation_and_rot:1,apply_translation_and_rot:2,ar:[1,2,3],area:1,around:2,arrai:[1,2],audio:5,augment:[1,2],axi:2,b:2,bar:1,base:[1,2,3],bbox1:1,bbox2:1,becaus:2,been:[1,2],befor:1,belong:1,best:1,between:2,bgr:1,bool:[1,2,3],both:1,bottom:1,bound:1,box1:1,box2:1,box:1,box_iou:1,boxes1:1,boxes2:1,build:[1,2],built:2,call:2,can:3,center:2,center_i:2,center_x:2,certain:2,chang:2,check:[1,2],checkpoint:1,clip:1,clip_coord:1,color:2,com:[1,2],combin:5,come:4,common:[0,2,3],comput:[1,2,3],compute_training_loss:1,compute_validation_loss:1,concept:5,confidence_threshold:3,contain:1,continu:1,continue_train:1,converg:2,convert:1,coord:1,coordin:2,correspond:[1,2],creat:[1,2],create_directory_if_does_not_exist:1,create_images_dataset_with_one_video:1,create_images_datasets_with_lpw_video:2,create_images_of_one_video_group:1,current:[1,2],custom:2,data:[1,2],dataload:1,dataset:[0,2,3],dataset_dir:1,datasetbuild:[0,2],defin:2,demonstr:5,detect:[2,3],develop:2,devic:[1,2,3],dimens:1,directli:2,directori:[1,2,3],disk:[1,2,3],displai:1,distanc:2,do_affine_grid_oper:1,doc:1,doe:1,domain:2,draw:2,draw_ellipse_on_imag:2,drawn:2,dure:1,each:[1,2],easili:2,element:1,ellips:[1,2],ellipse_height:2,ellipse_loss_funct:2,ellipse_util:0,ellipse_width:2,end:1,epoch:1,equat:2,euclidean:2,everi:1,exampl:2,execut:[1,2],exist:1,expect:1,extract:[1,2],ey:2,eye_track:0,eye_tracker_dir_path:2,eyetrackerdataset:0,eyetrackerdatasetbuild:0,eyetrackerdatasetbuilderofflinedataaugment:2,eyetrackerdatasetonlinedataaugment:2,eyetrackermodel:0,face:3,face_detect:3,face_detection_dir_path:3,facedetectiondataset:0,facedetectionmodel:0,facial:3,facial_detect:0,far:1,faster:2,file:1,file_nam:[1,2],five:2,follow:4,format:[1,2],forward:[2,3],fp:3,fpshelper:0,frame:[1,2,3],free:5,from:[1,2],functor:1,gener:2,get:[1,2,3,4],get_build:[1,2],get_from_list:2,get_from_opencv_ellips:2,get_image_and_label_on_disk:1,get_multiple_workers_safe_list_of_path:1,get_points_of_an_ellips:2,get_points_of_ellips:2,get_test_sub_dataset:[1,2,3],get_training_sub_dataset:[1,2,3],get_validation_sub_dataset:[1,2,3],getfp:3,gist:1,github:1,give:2,given:2,good:2,h:[1,2],handl:[1,2,3],happen:1,have:1,hear:5,height:[1,2,3],help:4,home:1,horizont:2,how:2,http:[1,2],idx:1,imag:[1,2,3],image_dimens:[1,2,3],image_tensor:1,image_util:0,images_dir:1,images_file_extens:1,img0_shap:1,img1_shap:1,img_shap:1,index:[1,5],info:1,inform:1,inherit:2,init:3,input:[2,3],input_image_height:[1,2],input_image_width:[1,2],instead:[1,2],intersect:1,intersection_over_union_threshold:3,inverse_norm:1,iou:1,its:[1,2],jaccard:1,k:2,know:1,label:[1,2,3],labels_dir:1,leak:1,learn:1,left:1,length:2,lie:2,limit:2,list:[1,2],load:1,load_best_model:1,load_model_and_training_info:1,log_nam:[1,2],longer:2,loss:[1,2],loss_funct:1,lpw:2,m:1,magnitud:1,main:[1,2],math:2,matrix:1,max:[1,2],mean:[1,2],memori:1,method:[1,2],metric:2,min:1,model:[1,2,3],model_dir_path:1,model_info_file_nam:1,modul:[1,2,3],more:2,mprostock:1,mseloss:2,multipl:1,must:[1,2],n:1,name:[1,2,3],nb_epoch:1,ndarrai:[1,3],necessari:1,need:[1,2],network:[1,2,3],neural:[1,2,3],nn:[2,3],none:1,normal:[1,2],normalizedellips:0,now:1,np:1,number:[1,2],number_of_point:2,numpi:[1,2],nx4:1,nxm:1,object:[1,2,3],offlin:2,offset:1,one:[1,2],onlin:2,open:5,opencv:[1,2,3],opencv_image_to_tensor:1,oper:[1,2],optim:1,option:[1,2],order:2,other:1,otherwis:2,ouput:2,output_dir_path:[1,2],output_image_tensor:[1,2],over:1,overwrit:2,overwritten:[1,2],pair:[1,2,3],pairwis:1,paralel:2,param:1,paramet:[1,2,3],parametr:2,parent:2,pars:2,parse_current_annot:2,part:3,pass:[1,2,3],path:1,per:1,perform:[1,2],phi:[1,2],pi:2,pixel:[1,2],plot:1,png:1,point:2,polar:2,posit:2,possibl:5,predict:[2,3],present:1,prevent:1,process:[1,2,5],process_fram:[1,2],process_image_label_pair:[1,2],processed_fram:2,progress:1,proof:5,pth:1,pupil:2,pytorch:[1,2,3],question:2,quintuplet:2,rad:2,radian:2,randomli:[1,2],rang:1,rate:1,rather:2,ratio_pad:1,rave:[1,2,3,4,5],reflect:2,rel:2,repres:[1,2],rescal:1,reset:3,respect:1,result:2,rgb:1,right:[1,3],root:1,root_dir_path:1,root_path:1,rooth_path:1,rotat:[1,2],rotate_around_image_cent:2,rotatio:2,rotation_angle_extremum:1,run:2,runner:1,s:2,save:[1,2],save_image_label_pair:1,save_model_and_training_info:1,saved_model:1,scale:1,scale_coord:1,scale_coords_landmark:1,schedul:1,select:1,serial:2,session:2,set:1,setfp:3,shape:[1,3],should:[1,2],show:1,sigmoid:2,smallest:1,so:1,some:[1,2],soon:4,sourc:5,source_dir:[1,2],specif:1,specifi:2,stackexchang:2,start:[3,4],std:1,stop:1,string:[1,2,3],sub:[1,2,3],sub_dataset_dir:[1,2,3],take:[1,2],target:2,tensor:[1,2,3],tensor_to_opencv_imag:1,termin:1,terminate_training_thread:1,test:[1,2,3],test_dir:1,them:[1,2],therot:1,theta:2,thi:[1,2],thick:2,thread:1,time:[1,3],to_list:2,todo:2,top:1,torch:[1,2,3],tracker:2,train:[1,2,3],train_with_valid:1,trainer:0,training_dir:1,training_load:1,training_mean:[1,2,3],training_sess:1,training_sessions_dir:1,training_std:[1,2,3],transform:[1,2],translat:[1,2],tupl:[1,2],type:[1,2,3],undo:1,union:1,unorm:1,updat:1,update_plot:1,us:[1,2,3],user:1,util:1,valid:[1,2,3],validation_dir:1,validation_load:1,valu:[1,2],version:1,vertic:2,video:[1,2,5],video_path:1,videos_dir:1,w:1,wa:[1,2],want:[1,3],we:3,weight:1,what:2,when:1,where:[1,2],whether:1,which:[1,2],width:[1,2,3],wish:1,work:1,worker:1,write:3,writefpstofram:3,x1:1,x2:1,x:[1,2,3],x_extremum:1,xy1:1,xy2:1,xywh2xyxi:1,xywh:1,xyxi:1,xyxy2xywh:1,y1:1,y2:1,y:[1,2],y_extremum:1,you:[1,4]},titles:["RAVE","common","eye_tracker","facial_detection","Examples","Home"],titleterms:{common:1,dataset:1,datasetbuild:1,ellipse_util:2,exampl:4,eye_track:2,eyetrackerdataset:2,eyetrackerdatasetbuild:2,eyetrackermodel:2,facedetectiondataset:3,facedetectionmodel:3,facial_detect:3,fpshelper:3,home:5,image_util:1,normalizedellips:2,rave:0,trainer:1}})