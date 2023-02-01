import sys
import csv
import glob

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from shapely import STRtree
from shapely.geometry import Polygon, Point
from matplotlib.collections import PolyCollection
from multiprocessing import Pool

from helpers import get_city_centroid, calculate_xgas, ds_groups, read


data_fields = {
    "CO": [
        "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/water_total_column",
        "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds",
        "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds",
        "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_pressure",
        "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude",
        "/PRODUCT/latitude",
        "/PRODUCT/longitude",
        "/PRODUCT/qa_value",
        "/PRODUCT/time_utc",
        "/PRODUCT/carbonmonoxide_total_column_corrected",
        # "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/column_averaging_kernel",
    ],
    "NO2": [
        "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/water_slant_column_density",
        "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds",
        "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds",
        "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_pressure",
        "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude",
        "/PRODUCT/SUPPORT_DATA/INPUT_DATA/northward_wind",
        "/PRODUCT/SUPPORT_DATA/INPUT_DATA/eastward_wind",
        "/PRODUCT/latitude",
        "/PRODUCT/longitude",
        "/PRODUCT/qa_value",
        "/PRODUCT/time_utc",
        "/PRODUCT/nitrogendioxide_tropospheric_column",
        # "/PRODUCT/air_mass_factor_troposphere",
        "/PRODUCT/air_mass_factor_total",
    ],
}

qa_values = {"CO": 0.5, "NO2": 0.75} #Readme for Co says 0.7 with ak


def get_files(year_start, month_start, year_end, month_end, product="CO"):

    result_list = []
    for year in range(int(year_start), int(year_end) + 1):
        year = str(year)
        for month in range(month_start, month_end + 1):
            month= str(month).zfill(2)

            for day in range(1, 32):
                day = str(day).zfill(2)

                date = year + month + day
                dir = f"/export/data2/scratch/tropomi/collection_03/{product}/"
                days = glob.glob(dir + "*_L2__CO_____" + date + "*.nc")

                if len(days) > 0:
                    result_list.append(days)
    return result_list


def get_other_file(floc):
    dir = floc.rpartition("/")[0]
    just_file = floc.split("/")[-1]
    product = just_file.split("_")[4]

    orbit_start_end = just_file[20:51]

    if product == "CO":
        other_dir = dir.replace("/CO", "/NO2")
        other_floc = other_dir + "/*" + orbit_start_end + "*.nc"

    elif product == "NO2":
        other_dir = dir.replace("/NO2", "/CO")
        other_floc = other_dir + "/*" + orbit_start_end + "*.nc"

    file_check = glob.glob(other_floc)

    if len(file_check) == 1:
        return file_check[0]
    elif len(file_check) > 1:
        raise FileExistsError("Two files found")
    elif len(file_check) == 0:
        #raise FileNotFoundError("No corresponding file found to: " + floc)
        print("No corresponding file found to: " + floc)
        return None

def open_file(floc):
    product = floc.split("/")[-1].split("_")[4]
    vgrps = ds_groups(data_fields[product])
    ds = read(floc, vgrps, dc_times=True)
    ds.attrs["product"] = product

    if product == "NO2":
        water_attrs = ds["water_slant_column_density"].attrs
        ds["water_total_column"] = (
            ds["water_slant_column_density"] / ds["air_mass_factor_total"]
        )
        ds["water_total_column"].attrs = water_attrs

    ds = ds.reset_coords(["latitude", "longitude"])
    ds["scanline"] = ds.scanline.astype(np.int32)
    ds["ground_pixel"] = ds.ground_pixel.astype(np.int32)
    ds = ds.squeeze("time").stack(sounding_dim=["scanline", "ground_pixel"])

    ds = ds.drop_vars(["scanline", "ground_pixel", "sounding_dim"])
    if "corner" in ds.dims:
        ds = ds.transpose("sounding_dim", "corner")

    ds = ds.assign_coords({"latitude": ds["latitude"], "longitude": ds["longitude"]})

    orbit = int(floc.split("/")[-1].split("_")[-4])
    orbit_da = xr.DataArray(np.full(len(ds.sounding_dim), orbit), dims="sounding_dim")
    
    ds = ds.assign_coords({"orbit": orbit_da})
    ds = ds.assign_coords({"corner": ds['corner']})
    
    return ds


def pixels_exist(floc, city_centroid, bound=1.0):
    product = floc.split("/")[-1].split("_")[4]

    vlst = [
        "PRODUCT/latitude",
        "PRODUCT/longitude",
        "PRODUCT/qa_value",
    ]

    clat, clon = city_centroid
    latmn = clat - bound
    lonmn = clon - bound
    latmx = clat + bound
    lonmx = clon + bound

    dat = read(floc, ds_groups(vlst), dc_times=False)

    lat_sel = (dat.latitude > latmn) & (dat.latitude < latmx)
    lon_sel = (dat.longitude > lonmn) & (dat.longitude < lonmx)

    qa_sel = dat.qa_value > qa_values[product]

    sel = lat_sel & lon_sel & qa_sel


    count=np.count_nonzero(sel)   

    return count > 10


def crop(ds, city_centroid, bound=1.0):
    clat, clon = city_centroid
    latmn = clat - bound
    lonmn = clon - bound
    latmx = clat + bound
    lonmx = clon + bound

    lat_sel = (ds.latitude >= latmn) & (ds.latitude <= latmx)
    lon_sel = (ds.longitude >= lonmn) & (ds.longitude <= lonmx)
    qa_sel = ds.qa_value >= qa_values[ds.attrs["product"]]

    sel = lat_sel & lon_sel & qa_sel
    ds = ds.where(sel, drop=True)

    return ds


def add_xgas(ds, add_water=False):
    product = ds.attrs["product"]

    if add_water:
        #Add XH2O basde on water total column from ds
        column = ds["water_total_column"].values
        standard_name = "water_vapor_dry_atmosphere_mole_fraction"
        long_name = "Water vapor column-averaged dry air mole fraction"
        product = "H2O"

    elif ds.attrs["product"] == "CO":
        column = ds["carbonmonoxide_total_column_corrected"].values
        standard_name = "carbonmonoxide_dry_atmosphere_mole_fraction"
        long_name = "Carbon monoxide column-averaged dry air mole fraction"

    elif ds.attrs["product"] == "NO2":
        column = ds["nitrogendioxide_tropospheric_column"].values
        standard_name = "nitrogendioxide_dry_atmosphere_mole_fraction"
        long_name = "Nitrogen dioxide column-averaged dry air mole fraction"

    xgas = calculate_xgas(
        column,
        ds["surface_pressure"].values,
        ds["latitude"].values,
        ds["water_total_column"].values,
    )

    ds[f"X{product}"] = ("sounding_dim", xgas)
    ds[f"X{product}"].attrs["standard_name"] = standard_name
    ds[f"X{product}"].attrs["long_name"] = long_name
    ds[f"X{product}"].attrs["units"] = "parts_per_billion_(1e-9)"

    return ds




def do(floc, other_floc, city_centroid, bound=1.0):
    ds1 = open_file(floc)
    ds2 = open_file(other_floc)

    ds1 = crop(ds1, city_centroid, bound)
    ds2 = crop(ds2, city_centroid, bound)
        
    if ds1.sounding_dim.size == 0 or ds2.sounding_dim.size == 0:
        return None
    ds1 = add_xgas(ds1)
    ds2 = add_xgas(ds2)

    if ds1.attrs["product"] == "CO":
        ds_co = ds1
        ds_no2 = ds2
    else:
        ds_co = ds2
        ds_no2 = ds1

    ds_no2 = add_xgas(ds_no2, add_water=True)

    pgons = [
        Polygon(zip(x, y))
        for x, y in zip(
            ds_co["longitude_bounds"].values, ds_co["latitude_bounds"].values
        )
    ]

    points = [
        Point(x, y)
        for x, y in zip(ds_no2["longitude"].values, ds_no2["latitude"].values)
    ]

    tree = STRtree(pgons)
    res = tree.query(points)
    res = [o for o in zip(res[0], res[1]) if pgons[o[1]].intersects(points[o[0]])]
    
    if len(res) == 0:
        return None

    res = list(zip(*res))  # res[0] contains input indices, res[1] contains tree indices

    ind_no2 = list(res[0])
    ind_co = list(res[1])

    ds_no2= ds_no2.rename({"latitude_bounds": "latitude_bounds_no2", "longitude_bounds": "longitude_bounds_no2"})
    ds_co = ds_co.rename({"latitude_bounds": "latitude_bounds_co", "longitude_bounds": "longitude_bounds_co"})

    ds_combined = xr.merge(
        [
            ds_no2["XNO2"][ind_no2],
            ds_no2["XH2O"][ind_no2],
            ds_no2["nitrogendioxide_tropospheric_column"][ind_no2],
            ds_no2["surface_pressure"][ind_no2],
            ds_no2["surface_altitude"][ind_no2],
            ds_no2["water_total_column"][ind_no2],
            ds_no2["time_utc"][ind_no2],
            ds_no2["northward_wind"][ind_no2],
            ds_no2["eastward_wind"][ind_no2],
            ds_no2["latitude_bounds_no2"][ind_no2],
            ds_no2["longitude_bounds_no2"][ind_no2],
            ds_co["XCO"][ind_co],
            ds_co["carbonmonoxide_total_column_corrected"][ind_co],
            ds_co["latitude_bounds_co"][ind_co],
            ds_co["longitude_bounds_co"][ind_co],
        ],
        compat="override",  # skip comparing and copy attrs from the first dataset to the result
    )

    ds_combined["time_utc"] = ds_combined["time_utc"].astype(str)

    ds_combined = ds_combined.rename({"time": "day"}).expand_dims("day")
    


    return ds_combined

    if False:
        plt.figure()
        for p in pgons:
            plt.plot(*p.exterior.xy, "b-")
        for point in points:
            plt.plot(*point.xy, "ro")

        plt.figure()
        for i, j in zip(res[0], res[1]):

            plt.plot(*points[i].xy, "ro")
            plt.fill(*pgons[j].exterior.xy, "b-")

    if False:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ds_first, ds_second = ds_co, ds_combined
        prod = "XCO"

        verts = [
            list(zip(x, y))
            for x, y in zip(
                ds_first["longitude_bounds"].values, ds_first["latitude_bounds"].values
            )
        ]

        col = PolyCollection(verts, array=ds_first[prod].values)
        ax1.add_collection(col)
        ax1.autoscale_view()
        fig.colorbar(col, ax=ax1)

        verts = [
            list(zip(x, y))
            for x, y in zip(
                ds_second["longitude_bounds"].values,
                ds_second["latitude_bounds"].values,
            )
        ]

        col = PolyCollection(verts, array=ds_second[prod].values)
        ax2.add_collection(col)
        ax2.autoscale_view()
        fig.colorbar(col, ax=ax2)


def combine_days(day_list):
    longest_sounding_dim = max([day.dims["sounding_dim"] for day in day_list])

    day_list_padded = []
    for day in day_list:
        ds_padded = day.pad(
            sounding_dim=(0, longest_sounding_dim - day.dims["sounding_dim"]),
            mode="constant",
            keep_attrs=True,
        )
        day_list_padded.append(ds_padded)

    ds_concat = xr.concat(day_list_padded, dim="day", combine_attrs="drop_conflicts")

    return ds_concat


def modify_attrs(ds):

    for var in ds.keys():
        # ds[var].encoding['_FillValue'] = None
        if "coordinates" in ds[var].attrs:
            del ds[var].attrs["coordinates"]

    del ds["nitrogendioxide_tropospheric_column"].attrs["ancillary_variables"]
    del ds["carbonmonoxide_total_column_corrected"].attrs["ancillary_variables"]
    del ds["water_total_column"].attrs["ancillary_variables"]

    ds["orbit"].attrs["long_name"] = "Orbit number. One or two per day"
    ds["water_total_column"].attrs["long_name"] = "Water vapor total column density"
    ds["water_total_column"].attrs["units"] = "mol m-2"
    
    ds["latitude"].attrs["long_name"] = "pixel (no2) center latitude"
    ds["longitude"].attrs["long_name"] = "pixel (no2) center longitude"

def do_full_day(day_list):
    list_of_orbit_ds = []
    for orbit_file in day_list:
        exist = pixels_exist(orbit_file, city_centroid, bound=bound)

        if exist:
            other_orbit_file = get_other_file(orbit_file)
            
            if other_orbit_file is not None:
                orbit_ds = do(orbit_file, other_orbit_file, city_centroid, bound=bound)

                if orbit_ds is not None:
                    list_of_orbit_ds.append(orbit_ds)

    if len(list_of_orbit_ds) == 2:
        return xr.concat(list_of_orbit_ds, dim="sounding_dim", combine_attrs="drop_conflicts")

    elif len(list_of_orbit_ds) == 1:
        return list_of_orbit_ds[0]

    elif len(list_of_orbit_ds) == 0:
        return None
    
    else:
        raise ValueError("Something went wrong")


city="NewYork-Newark" #2.5
#city="Phoenix-Mesa" #2.0
y_start, m_start, y_end, m_end = 2022, 7, 2022, 12
bound=2.5

city_centroid = get_city_centroid(city)

month_list = get_files(y_start, m_start, y_end, m_end)

print("Processing each day...")

pool=Pool(20)
results=pool.map(do_full_day, month_list)
list_of_day_ds=[x for x in results if x is not None]

print("Combining days...")
ds_result = combine_days(list_of_day_ds)
modify_attrs(ds_result)

m_start=str(m_start).zfill(2)
m_end=str(m_end).zfill(2)

print("Saving to netcdf4...")
#ds_result.to_netcdf(f"{city}_{y_start}{m_start}_{y_end}{m_end}.nc")
#ds_result.load().to_netcdf(f"{city}_{y_start}{m_start}_{y_end}{m_end}.nc")
