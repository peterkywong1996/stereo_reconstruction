export PYTHONPATH=/usr/local/lib/python3.6/dist-packages:$PYTHON_PATH

trap "exit 0" SIGINT SIGTERM

[ $1 ] || { echo "Enter start id"; exit 1; }
RID=$1
for pm in bf flann; do
  for mi in 25000; do
    for dt in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4; do
      for me in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4; do
	for mcd in 10 15 20; do
python3 main.py \
--dir_img1=./data/non-rectified/01/im0.png \
--dir_img2=./data/non-rectified/01/im1.png \
--point_matcher=$pm \
--filter_points \
--dist_threshold=$dt \
--use_ransac \
--min_data=8 \
--min_close_data=$mcd \
--max_iteration=$mi \
--max_error=$me \
--save_result \
--result_id=$RID
RID=$(expr $RID + 1)
done
done
done
done
done
#--use_manual_baseline \
#--dir_x1=data/points/01_x1.txt \
#--dir_x2=data/points/02_x2.txt \
