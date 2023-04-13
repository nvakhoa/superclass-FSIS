export CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2



SHOTS=(1 5 10)
SHOTS=(1)
for shot in ${SHOTS[*]}
do
    MODEL_NAME=SMS_LR_${shot}shot

    OUTPUT_DIR=checkpoints/main_results

    config=configs/ours/fs/${MODEL_NAME}.yaml
    NOVEL_WEIGHT=checkpoints/pretrained_weights/${MODEL_NAME}.pth

    python tools/run_train.py --num-gpus ${NGPUS} \
                                --dist-url auto \
                                --config-file ${config} \
                                --method 'fine-tuning' \
                                --eval-only \
                                 --opts MODEL.WEIGHTS ${NOVEL_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} MODEL.ROI_HEADS.OUTPUT_LAYER 'SMS_LR' MODEL.ROI_HEADS.NAME 'StandardSuperClassROIHead'


done

