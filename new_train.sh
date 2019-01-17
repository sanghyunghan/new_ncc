#제일 처음 full training 한거, resnet101, ecpoch 50, 기타 learning rate, segmentation, negative & positive overlap 은 이전에 쓰던거 그대로, data만 normal data 추가함
#nohup python -u keras_retinanet/bin/new_train.py csv /home/huray/data/new_trainset/0830_new_train_data.csv /home/huray/data/new_trainset/class_malig_and_norm.csv  --evaluation True --val-annotations /home/huray/data/new_trainset/0830_new_valid_data.csv &
#두번째 full training... resnet50, epoch(100), data는 NCC test용 데이터 제외
# learning rate(1e-5), segmentation, negative(0.4) & positive(0.5) overlap 등은 retinanet 처음 기본값으로 되돌려 수행
#nohup python -u keras_retinanet/bin/new_train.py csv /home/huray/data/new_trainset/0907_new_train_data7072_with_only_normal_NCC_trainset.csv /home/huray/data/new_trainset/class_malig_and_norm.csv  --evaluation True --val-annotations /home/huray/data/new_trainset/0907_new_valid_data600_with_only_normal_NCC_trainset.csv &
#세번째 no from snapshot full training... resnet50, epoch(60), data는 NCC test용 데이터 제외
#learning rate(1e-5), segmentation(min(0.7) & max(1.3) scaling)) , negative(0.5) & positive(0.5) overlap
#nohup python -u keras_retinanet/bin/new_train.py csv /home/huray/data/new_trainset/0907_new_train_data7072_with_only_normal_NCC_trainset.csv /home/huray/data/new_trainset/class_malig_and_norm.csv  --evaluation True --val-annotations /home/huray/data/new_trainset/0907_new_valid_data600_with_only_normal_NCC_trainset.csv &
#네번째 abnormal 데이터만(NCC test abnormal 데이터 제외, valid 데이터로 활용) training... resnet152, epoch(100)
#learning rate(1e-5), segmentation(min(0.7) & max(1.3) scaling)) , negative(0.5) & positive(0.5) overlap
nohup python -u keras_retinanet/bin/new_train.py csv /home/huray/data/new_trainset/0919_new_train_data_only_abnormal_exclude_NCC_test.csv /home/huray/data/new_trainset/class_2_malig_or_norm.csv  --evaluation True --val-annotations /home/huray/data/new_trainset/0919_new_valid_data_NCC_test_only_anno.csv &