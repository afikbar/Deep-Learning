#!/bin/sh

# questions
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip

# answers
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip

# bottom up features (https://github.com/peteanderson80/bottom-up-attention)
wget https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip https://imagecaption.blob.core.windows.net/imagecaption/test2015_36.zip

unzip "*.zip"
rm *.zip
