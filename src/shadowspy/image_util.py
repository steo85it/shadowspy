def read_img_properties(cumindex, columns=['PRODUCT_ID', 'START_TIME']):

    cumindex_red = cumindex[columns]
    cumindex_red.loc[:, 'PRODUCT_ID'] = cumindex_red['PRODUCT_ID'].str.strip()

    return cumindex_red
