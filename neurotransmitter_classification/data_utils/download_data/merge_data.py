import pandas as pd
import glob

gt_dir = "/media/samia/DATA/mounts/cephfs/catena/helpers/neurotransmitter/202508-hemibrain_gt_data/positives_bbox-08/"

list_of_csvs = glob.glob(f"{gt_dir}/*.csv")

temp_df = pd.DataFrame()
for csv in list_of_csvs:
    file_ = pd.read_csv(csv)
    temp_df = pd.concat([temp_df, file_], ignore_index=True)

temp_df.to_csv(f"{gt_dir}/202508-hemibrain_gt_data_all_data.csv", index=False)



