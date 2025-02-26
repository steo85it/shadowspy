def read_img_properties(imgs, cumindex):

    cumindex_red = cumindex[['PRODUCT_ID', 'START_TIME']]
    cumindex_red = cumindex_red.loc[cumindex.PRODUCT_ID.str.strip().isin(imgs)].reset_index(drop=True)
    cumindex_red.PRODUCT_ID = cumindex_red.PRODUCT_ID.str.strip()

    return cumindex_red
