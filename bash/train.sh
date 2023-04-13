
export CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2

config=configs/ours/mask_rcnn_R_101_FPN_base_220k.yaml

# pre-trained Resnet-101
WEIGHTS=weights/R-101.pkl

python tools/run_train.py --num-gpus ${NGPUS} --dist-url auto --resume --config-file ${config} --opts SOLVER.CHECKPOINT_PERIOD 50000 MODEL.WEIGHTS ${WEIGHTS}

OUTPUT_DIR=checkpoints/BaseModel_MRCNN_R_101_FPN
WEIGHT=${OUTPUT_DIR}/model_final.pth
NEW_BASE_WEIGHT=${OUTPUT_DIR}/model_reset_ckpt.pth
if [ -f "$NEW_BASE_WEIGHT" ]; then
    echo "$NEW_BASE_WEIGHT exists."
else 
    echo "$NEW_BASE_WEIGHT does not exist."
    python tools/reset_ckpt.py --src ${WEIGHT} --save-name ${NEW_BASE_WEIGHT}
fi


METHODS=("superclass")
PHASES=(all)
SHOTS=(1 5 10)
SHOTS=(1)
for method in ${METHODS[*]}
do
for phase in ${PHASES[*]}
do
for shot in ${SHOTS[*]}
do

    config=configs/ours/mask_rcnn_R_101_FPN_fullclsag_fine_tuning_${phase}_${shot}shot_superclass.yaml
  
    OUTPUT_DIR=checkpoints/${method}/SMS+LR_${phase}_${shot}shot

    NEW_SUPER_CLASS_CONFIG=configs/superclass/mask_rcnn_R_101_FPN_${phase}_${shot}shot_train_test_super-class.yaml

    NOVEL_WEIGHT=${OUTPUT_DIR}/model_init.pth

    python tools/run_train.py --num-gpus 1 \
                                --dist-url auto \
                                --config-file ${config} \
                                --method ${method} \
                                --src1 ${NEW_SUPER_CLASS_CONFIG} \
                                --opts MODEL.WEIGHTS ${NEW_BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}  ${cfg_MODEL}

    method2='fine-tuning'
    python tools/run_train.py --num-gpus ${NGPUS} \
                                --dist-url auto \
                                --config-file ${NEW_SUPER_CLASS_CONFIG} \
                                --method ${method2} \
                                --opts MODEL.WEIGHTS ${NOVEL_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}


done
done
done

