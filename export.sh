# -c Set the training algorithm yml configuration file
# -o Set optional parameters
# Global.pretrained_model parameter Set the training model address to be converted without adding the file suffix .pdmodel, .pdopt or .pdparams.
# Global.save_inference_dir Set the address where the converted model will be saved.

python tools/export_model.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o G lobal.pretrained_model=output/en_PP-OCRv3_rec_train/best_accuracy  Global.save_inference_dir=output/inference/en_PP-OCRv3_rec/