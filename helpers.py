import csv
import numpy as np
import xarray as xr

from ambiance import Atmosphere
from numpy.lib.arraysetops import unique


def get_city_centroid(name):
    csv_floc = "cities.csv"
    with open(csv_floc, "r") as csv_in:
        rows = csv.reader(csv_in)
        for k, row in enumerate(rows):
            if k == 0:
                attrs = [a.lower() for a in row]
                name_ind = attrs.index("name")
            else:
                if row[name_ind] == name:
                    bbox = [row[1], row[2], row[3], row[4]]
                    # bbox = np.array([self.bbx_latmn, self.bbx_lonmn, self.bbx_latmx, self.bbx_lonmx])
                    centroid = [float(row[5]), float(row[6])]
                    # centroid = np.array([self.gcpnt_lat, self.gcpnt_lon])
                    return centroid


def calculate_g_air(surface_pressure: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """Calculate pressure-weighted gravity for given surface pressures and latitudes."""

    # One scale height is at 1/e of the surface pressure
    altitude_H = Atmosphere.from_pressure(surface_pressure / np.e).h

    if not surface_pressure.shape == latitude.shape:
        print("waht")
        if surface_pressure.ndim == 2 and latitude.ndim == 1:
            latmat = np.tile(latitude, (surface_pressure.shape[1], 1)).T
        else:
            raise ValueError("Surface pressure and latitude must have the same shape.")
    else:
        latmat = latitude

    altmat = altitude_H

    gm = 3.9862216e14  # Gravitational constant times Earth's Mass (m3/s2) in GFIT
    omega = 7.292116e-05  # Earth's angular rotational velocity (radians/s) from GFIT
    con = 0.006738  # (a/b)**2-1 where a & b are equatorial & polar radii from GFIT
    shc = 1.6235e-03  # 2nd harmonic coefficient of Earth's gravity field from GFIT
    eqrad = 6378178  # Equatorial Radius (m) from GFIT

    gclat = np.arctan(np.tan(latmat * np.pi / 180) / (1 + con))

    radius = altmat + eqrad / np.sqrt(1 + con * np.sin(gclat) ** 2)
    ff = (radius / eqrad) ** 2
    hh = radius * omega**2
    ge = gm / (eqrad**2)  # = gravity at Re

    g = (
        ge * (1 - shc * (3 * np.sin(gclat) ** 2 - 1) / ff) / ff
        - hh * np.cos(gclat) ** 2
    ) * (1 + 0.5 * (np.sin(gclat) * np.cos(gclat) * (hh / ge + 2 * shc / ff**2)) ** 2)

    return g


def calculate_xgas(column_amount, surface_pressure, latitude, h2o_column_amount):
    """Calculate X_gas."""

    m_dry_air = 0.0289644  # kg/mol
    m_h2o = 0.01801528  # kg/mol

    g_air = calculate_g_air(surface_pressure, latitude)

    column_dry_air = (
        surface_pressure / (g_air * m_dry_air) - h2o_column_amount * m_h2o / m_dry_air
    )

    xgas = column_amount / column_dry_air * 1e9

    return xgas


def ds_groups(keys):
    """
    Get the unique set of
    netCDF groups that should
    be loaded in.

    args:
        keys: list of netCDF
        variable paths to be
        loaded.
    """
    grps = ["/".join(k.split("/")[:-1]) for k in keys]
    ugrps = unique(grps)

    vgrps = dict(zip(list(ugrps), [[] for _ in range(len(ugrps))]))
    for key in keys:
        k = "/".join(key.split("/")[:-1])
        var = key.split("/")[-1]
        vgrps[k].append(var)

    return vgrps


def read(floc, vgrps, dc_times=False):

    xds = []
    for grp, vsel in vgrps.items():
        with xr.open_dataset(floc, group=grp, chunks={}, decode_times=dc_times) as ods:
            ods = ods[vsel]
            xds.append(ods)

    dat = xr.merge(xds)
    return dat


if __name__ == "__main__":
    import xarray as xr
    import matplotlib.pyplot as plt

    a = xr.open_dataset("Houston_202207_202212.nc")

    print(a)

    if False:
        for i in [1, 5, 10]:
            plt.figure()
            b = a.isel(day=i)
            print(b["carbonmonoxide_total_column_corrected"].values)
            b.plot.scatter(
                x="longitude",
                y="latitude",
                hue="carbonmonoxide_total_column_corrected",
                cmap="viridis",
            )
