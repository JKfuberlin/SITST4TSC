# %%
import geopandas as gpd
import os
import utils.csv as csv

# %%
SHP_DIR = 'D:\\Deutschland\\FUB\\master_thesis\\data\\Reference_data\\bw_all_polygons'
OUTPUT_DIR = 'D:\\Deutschland\\FUB\\master_thesis\\data\\ref\\all'
INPUT_SHP = 'buffered_wgs_bw_polygons.shp'
OUTPUT_SHP = 'bw_polygons_pure.shp'
REF_CSV = 'reference_pure.my_csv'

# %%
def load_shp_file() -> gpd.GeoDataFrame:
    in_path = os.path.join(SHP_DIR, INPUT_SHP)
    gdf = gpd.read_file(in_path)
    print(f'import file {in_path}')
    return gdf

# %%
def export_shp_file(gdf:gpd.GeoDataFrame) -> None:
    out_path = os.path.join(SHP_DIR, OUTPUT_SHP)
    gpd.GeoDataFrame.to_file(gdf, out_path)
    print(f'export file {out_path}')

# %%
def shuffle(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # remove useless columns
    keys = ['OBJECTID', 'KATEGORIE', 'OBJEKTART', 'NUART', 'WEFLKZ', 'FBEZ', 'BETR', 'REVIER', 'DIST', 'ABT', 
    'RWET', 'BI', 'AI_FOLGE', 'BEST_BEZ', 'STICHTAG', 'LWET', 'ENTSTEHUNG', 'ENTSTEHU_1', 'AENDERUNG_', 
    'AENDERUNG1', 'LOCK__ID', 'GUID', 'GUID_ABT', 'GUID_DIS', 'GUID_BTR', 'GUID_REV', 'GUID_FBZ', 'GUID_NACHF', 
    'SFL_BEZ', 'GEFUEGE_BE', 'MASSN_BEZ', 'BRUCHBESTA', 'FEVERFAHRE', 'TURNUS', 'LWET_DARST', 'DAUERWALD', 
    'ALTKL_HB_A', 'ALTKL_IT_A', 'ALTKL_HB_B', 'BAA_PKT_TE', 'BAA_PKT_BU', 'BAA_PKT_EI', 'BAA_PKT_BL', 
    'BAA_PKT_PA', 'BAA_PKT_FI', 'BAA_PKT_TA', 'BAA_PKT_DG', 'BAA_PKT_KI', 'BAA_PKT_LA', 'BAA_PKT_NB', 
    'BAA_PKT_LB', 'GD_AB', 'GD_BIS', 'FATID', 'FOKUS_ID', 'GEFUEGE', 'BEST_BEZ1', 'BEST_BEZ2', 'BEST_BEZ3', 
    'BESTTYP', 'BU_WLRT', 'LWET_TEXT', 'MASSNAHMEN', 'NHB_BEZ', 'ALT_HB', 'ALT_IT', 'BST_ART_ID', 'NWW_BHT', 
    'NWW_KAT', 'SHAPE_STAr', 'SHAPE_STLe',]
    gdf.drop(columns=keys, inplace=True)
    # remove useless rows
    gdf = gdf.drop(gdf[gdf['BST1_BA_1'] == 0].index)
    gdf = gdf.drop(gdf[gdf['BST2_BA_1'] != 0].index)
    gdf = gdf.drop(gdf[gdf['BST3_BA_1'] != 0].index)
    # add uuid to each polygon
    gdf['id'] = gdf.index + 1
    return gdf

# %%
def buffer(polygons:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # buffer
    polygons["geometry"] = gpd.GeoDataFrame.buffer(polygons, -10)
    print("Buffer -10 m")
    # reproject
    polygons = polygons.to_crs(epsg=4326)
    print("Reproject to EPSG:4326")
    return polygons

# %%
def select_pure(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Selcet pure classes whose main species is >= 90%"""
    gdf = gdf.drop(gdf[gdf['BST1_BAA_1'] < 90].index)
    return gdf

# %%
def export_csv_reference(gdf:gpd.GeoDataFrame) -> None:
    # delete unused columns
    cols = ['BST2_BA_1', 'BST2_BA_2', 'BST2_BA_3', 'BST2_BA_4', 'BST2_BA_5', 'BST2_BA_6', 'BST2_BA_7', 'BST2_BA_8', 
        'BST2_BAA_1', 'BST2_BAA_2', 'BST2_BAA_3', 'BST2_BAA_4', 'BST2_BAA_5', 'BST2_BAA_6', 'BST2_BAA_7', 'BST2_BAA_8', 
        'BST3_BA_1', 'BST3_BA_2', 'BST3_BA_3', 'BST3_BA_4', 'BST3_BA_5', 'BST3_BA_6', 'BST3_BA_7', 'BST3_BA_8', 
        'BST3_BAA_1', 'BST3_BAA_2', 'BST3_BAA_3', 'BST3_BAA_4', 'BST3_BAA_5', 'BST3_BAA_6', 'BST3_BAA_7', 'BST3_BAA_8',
        'geometry']
    gdf.drop(columns=cols, inplace=True)
    # export result as my_csv file
    ref_path = os.path.join(OUTPUT_DIR, REF_CSV)
    csv.export(gdf, ref_path, False)

# %%
if __name__ == "__main__":
    polygons = load_shp_file()
    # # load raw shp, clean, add index, buffer and reproject
    # polygons = shuffle(polygons)
    # buffered_polygons = buffer(polygons)
    # export_shp_file(buffered_polygons)
    # export_csv_reference(buffered_polygons)
    # select pure polygons
    pure_polygons = select_pure(polygons)
    export_shp_file(pure_polygons)
    export_csv_reference(pure_polygons)