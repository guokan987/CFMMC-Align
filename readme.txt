1 Download glove pretrained 300d word vectors to /data/glove/ and process it into a pickle file.
python txt2pickle.py

2 reprocess train/test questions:
python 1_preprocess_questions_oie_CMCIR.py --mode train
    
python 1_preprocess_questions_oie_CMCIR.py --mode test


3 To extract appearance feature with Swin or Resnet101 model:
Download Swin pretrained model (swin_large_patch4_window7_224_22k.pth) and place it to configs/.

python 1_preprocess_features_appearance_sutd_swin.py --model Swin --question_type none

4 To extract motion feature with Swin and Download Swin3D pretrained model (swin_base_patch244_window877_kinetics600_22k.pth) and place it to configs/.

python 1_preprocess_features_motion_sutd_swin.py --model Swin --question_type none


5 Train and Test

python train_SUTD_CTVQA_clip.py or python train_SUTD_CTVQA_clip.py 