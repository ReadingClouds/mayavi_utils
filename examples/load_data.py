import xarray as xr

from pathlib import Path
from monc_utils.data_utils.string_utils import get_string_index


def sortkey(p):
    idx_string = p.stem.split("_")[-1]
    return int(idx_string)

def preprocess(ds):
    if "options_database" in list(ds.data_vars):
        ds = ds.drop_vars(["options_database"])
    return ds

def load_data(data_path, file_prefix, ref_file, selector):
    files = sorted(data_path.glob(f"{file_prefix}{selector}.nc"), 
               key=sortkey)
               
    # print(list(files))
               
    ds_in = xr.open_mfdataset(files,
                              preprocess=preprocess,
                              combine_attrs="override")

    print(ds_in)


    dsod = xr.open_dataset(list(files)[0])
    for var in ["options_database", "z", "zn"]:
        ds_in[var] = dsod[var]
        
    dsod.close()
    

    ref_file = list(sorted(data_path.glob(f"{ref_file}{selector}.nc"), 
                           key=sortkey))[0]

    # print(ref_file)

    if ref_file is not None:
        ref_dataset = xr.open_dataset(ref_file)
    
        ref_dataset = ref_dataset[['prefn', 'rho', 'rhon', 'thref', ]]
        [itime] = get_string_index(ref_dataset.dims, ['time', ])
        timevar = list(ref_dataset.dims)[itime]
        
        ref_dataset = ref_dataset.isel({timevar:0}).squeeze(drop=True).drop(timevar)
    
        print(ref_dataset)
    
        dataset = xr.merge([ds_in, ref_dataset])
    else:
        ref_dataset = None
        dataset = ds_in
        
    return dataset
    
