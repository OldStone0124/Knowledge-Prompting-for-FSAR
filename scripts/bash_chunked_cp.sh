arr=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9")
 
for value in ${arr[@]}
do
  tmp_dir=/media/hdd/sda1/linhanxi/data/ss_v2/extracted_frms_OTAM/ssv2_256x256q5/train/train$value"*"/*
  cp -r $tmp_dir /media/hdd/sda1/linhanxi/data/ss_v2/extracted_frms_OTAM/ssv2_256x256q5_extraVal/
done
