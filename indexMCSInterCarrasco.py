import pygrib
import xarray as xr
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def read_grib_data(grib_file, indices, variable_mapping):
    grbs = pygrib.open(grib_file)
    ds = xr.Dataset()
    
    for idx in indices:
        grb = grbs.message(idx)
        variable_short_name = variable_mapping.get(idx, grb.shortName)
        unit_of_measure = grb.units
        
        latitudes, longitudes = grb.latlons()
        values = grb.values
        
        data = xr.DataArray(values, coords={'latitude': latitudes[:, 0], 'longitude': longitudes[0, :]},
                            dims=['latitude', 'longitude'], attrs={'units': unit_of_measure, 'long_name': grb.name, 'short_name': variable_short_name})
        ds[variable_short_name] = data
    
    grbs.close()
    ds.metpy.parse_cf()
    return ds

def calculate_mcs_index(ds):
    u_500hPa = ds.u500hPa
    v_500hPa = ds.v500hPa
    u_1000hPa = ds.u1000hPa
    v_1000hPa = ds.v1000hPa
    u_775hPa = ds.u775hPa
    v_775hPa = ds.v775hPa
    t_775hPa = ds.t775hPa
    w_800hPa = ds.w
    pli = ds.lftx4
    pwat = ds.pwatcapa
    
    mag500 = mpcalc.wind_speed(u_500hPa, v_500hPa)
    mag1000 = mpcalc.wind_speed(u_1000hPa, v_1000hPa)
    shear = mag500 - mag1000
    gradiente_horizontal = mpcalc.advection(t_775hPa, u_775hPa, v_775hPa)
    
    mcs = ((shear.values - 13.66) / 5.5) + ((gradiente_horizontal.values - 4.28e-5) / 5.19e-5) + (-(w_800hPa + 0.269) / 0.286) + (-(pli + 2.142) / 2.175)
    return mcs

# Interpolación de los datos de mcs para obtener el valor más cercano a Carrasco
def interp_avance(mcs):
    da = xr.DataArray(mcs[242:252,600:610].values.reshape(10, 10),
        [("x", mcs.longitude[600:610].values), ("y", mcs.latitude[242:252].values)],
    )
    longi = xr.DataArray([303.55, 303.65,303.75,303.85,303.95, 304.0,304.1,304.2], dims="z")
    latj = xr.DataArray([-34.55,-34.65,-34.75,-34.85,-34.95,-35.0,-35.1,-35.2], dims="z")
    daint = da.interp(x=longi, y=latj)
    inter_carr = daint[5].values
    print(daint[5].values)
    return inter_carr

def create_mcs_map(ds, mcs, inter_carr, pwat, pli):
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    extent = [-75, -45, -40, -16]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_title("SA-MCS Index", fontsize=16, fontweight='bold')
    
    masked_mcs_pwat = np.where((pwat < 27) & (pli > 0), np.nan, mcs)
    
    cmap = (mpl.colors.ListedColormap(['green', 'yellow', 'orange'])
            .with_extremes(over='red', under='white'))
    contourf_mcs = ax.contourf(ds['longitude'], ds['latitude'], masked_mcs_pwat, levels=[-1.15, -0.12, 1.58, 2.74], cmap=cmap, transform=ccrs.PlateCarree(),extend='both')
    
    bounds = [-1.15, -0.12, 1.58, 2.74]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cbar_temp = plt.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=ax,
        extend='both',
        extendfrac='auto',
        ticks=bounds,
        spacing='uniform',
        orientation='horizontal',
        label='SA-MCS Index',
        aspect=30,
        pad=0.02,
        shrink=0.8,
        fraction=0.1,
        drawedges=True
    )
    cbar_temp.set_ticklabels([-1.15, -0.12, 1.58, '2.74'])
    cbar_temp.set_label('SA-MCS Index', fontsize=14)
    
    # Agregar los valores de la variable en el mapa
    ax.text(-56.01, -33.83, "Carrasco", transform=ccrs.PlateCarree(), fontsize=12)
    ax.text(-56.01, -34.83, round(float(inter_carr),2), transform=ccrs.PlateCarree(), fontsize=10, 
            color='k', fontweight='bold', bbox=dict(facecolor='white', alpha=0.4, edgecolor='k', boxstyle='circle,pad=0.1'))
    
    # print(mcs.sel(latitude=-34.83, longitude=304.0, method='nearest').values) 
    plt.show()

def main():
    grib_file = "cdas1_2022011700_.t00z.pgrbh00.grib2"
    indices = [212, 213, 268, 272, 273, 281, 362, 363, 419, 420, 421, 611]
    variable_mapping = {
        212: 'u500hPa',
        213: 'v500hPa',
        268: 't775hPa',
        272: 'u775hPa',
        273: 'v775hPa',
        362: 'u1000hPa',
        363: 'v1000hPa',
        419: 'pwat30hPa',
        611: 'pwatcapa',
        421: 'lftx4',
    }

    ds = read_grib_data(grib_file, indices, variable_mapping)
    mcs = calculate_mcs_index(ds)
    inter_carr = interp_avance(mcs)
    create_mcs_map(ds, mcs, inter_carr, ds.pwatcapa, ds.lftx4)

if __name__ == "__main__":
    main()
