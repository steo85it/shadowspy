import logging
import datetime
import os.path

import pandas as pd

from shadowspy.image_util import read_img_properties


def fetch_and_process_data(opt):

    use_azi_ele = False
    use_image_times = False
    if opt.azi_ele_path not in [None, 'None']:
        use_azi_ele = True

        if not opt.point_source:
            logging.error("* Can only provide azimuth&elevation when using a point source.")
            exit()

        data_list = pd.read_csv(opt.azi_ele_path, sep=',', dtype='int').values.tolist()
        logging.info("- Computing illumination from azi/ele:", data_list)

    elif opt.images_index not in [None, 'None']:
        use_image_times = True
        images_index = opt.images_index
        cumindex = pd.read_csv(images_index, index_col=None)
        # get list of images from cumindex
        try:
            data_list = read_img_properties(cumindex, columns=['PRODUCT_ID', 'START_TIME', 'ortho_path'])
        except:
            full_index = pd.read_parquet("/home/tmckenna/nobackup/sfs_helper/examples/HLS/A3BA/root/CUMINDEX.parquet")
            img_list = [x for x in cumindex.img_name.values
                        if os.path.exists(f"/home/tmckenna/nobackup/sfs_helper/examples/HLS/A3BA/proc/tile_0/sel_0/prj/ba_align/{x}_map.tif")]
            cumindex = full_index.copy()
            cumindex = cumindex.loc[full_index.PRODUCT_ID.str.strip().isin(img_list)]
            # cumindex['ortho_path'] = [f"/home/tmckenna/nobackup/sfs_helper/examples/HLS/A3BA/proc/tile_0/sel_0/prj/ba_align_unselected/{img}_map.tif"
            #                           for img in cumindex.PRODUCT_ID.str.strip().values]
            cumindex.loc[:, 'ortho_path'] = [
                f"/home/tmckenna/nobackup/sfs_helper/examples/HLS/A3BA/proc/tile_0/sel_0/prj/ba_align/{img}_map.tif"
                for img in cumindex.PRODUCT_ID.str.strip().values]
            data_list = read_img_properties(cumindex, columns=['PRODUCT_ID', 'START_TIME', 'ortho_path'])

        data_list = [(row.iloc[0], row.iloc[1].strip(), row.iloc[2]) for idx, row in data_list.iterrows()]

    elif len(opt.epos_utc) > 0:
        data_list = opt.epos_utc

    else:
        start_time = datetime.datetime.strptime(opt.start_time, '%Y-%m-%d %H:%M:%S.%f')
        end_time = datetime.datetime.strptime(opt.end_time, '%Y-%m-%d %H:%M:%S.%f')
        s = pd.Series(pd.date_range(start_time, end_time, freq=f'{opt.time_step_hours}H')
                      .strftime('%Y-%m-%d %H:%M:%S.%f'))
        data_list = s.values.tolist()

    return data_list, use_azi_ele, use_image_times
