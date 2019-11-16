import netCDF4 as nc
import numpy as np
import zlib


def writenc(name, data):
    dataset = nc.Dataset(name, 'w', format='NETCDF4', zlib=True)
    keys = ['time', 'lat', 'lon', 'Total_precipitation_surface']
    dataset.createDimension(keys[0], len(data[0]))
    dataset.createVariable(keys[0], np.float64, (keys[0]), zlib=True)
    dataset.variables[keys[0]][:] = data[0]
    for i in range(1, len(keys) - 1):
        dataset.createDimension(keys[i], len(data[i]))
        dataset.createVariable(keys[i], np.float32, (keys[i]), zlib=True)
        dataset.variables[keys[i]][:] = data[i]
    dataset.createVariable(keys[-1], np.float32, tuple(keys[:-1]), zlib=True)
    d = np.array(data[-1])
    # print(d.shape)
    d = np.where(d < 0.1, 0, d)
    # data[-1] = data[-1].round(2)
    dataset.variables[keys[-1]][:] = d  # np.maximum(data[-1],0)
    tt = name.split('_')[-2]
    yy, mm, dd, hh = tt[0:4], tt[4:6], tt[6:8], tt[8:10]
    time = dataset.variables['time']
    time.units = 'minutes since %s-%s-%s %s:00:00' % (yy, mm, dd, hh)
    time.calendar = 'gregorian'
    dataset.variables['lat'].units = 'degrees_north'
    dataset.variables['lon'].units = 'degrees_east'
    dataset.variables['Total_precipitation_surface'].units = 'mm'
    dataset.close()
