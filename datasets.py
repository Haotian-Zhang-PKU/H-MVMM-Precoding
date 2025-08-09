import torch
import numpy as np
from torch.utils.data import Dataset

class Dataset_3MMFF(Dataset):
    def __init__(self, train_y_input, train_h_input, train_gps_input, train_rgb_input, train_lidar_input, num_pilot):
        self.ydata = train_y_input
        self.hdata = train_h_input
        self.gpsdata = train_gps_input
        self.rgbdata = train_rgb_input
        self.lidardata = train_lidar_input
        self.num_pilot = num_pilot
    def __len__(self):
        return len(self.ydata[0])  

    def __getitem__(self, item):
        user = len(self.ydata)
        ysamples = []
        for data in self.ydata:  
            sample = torch.from_numpy(data[item])
            sample_pilot = [sample[:,int(i)] for i in range(self.num_pilot)]  
            sample_pilot = torch.tensor(sample_pilot)
            sample_pilot_real = sample_pilot.real
            sample_pilot_imag = sample_pilot.imag
            sample_pilot = torch.cat([sample_pilot_real, sample_pilot_imag], dim=0)
            ysamples.append(sample_pilot)
              
        hsamples = []
        for data in self.hdata:   
            sample = torch.from_numpy(data[item])
            sample_real = sample.real
            sample_imag = sample.imag
            sample = torch.cat([sample_real, sample_imag], dim=0)
            hsamples.append(sample)
        
        gpssamples = []
        for data in self.gpsdata: 
            sample = torch.from_numpy(data[item])
            gpssamples.append(sample)
        
        RGBsamples = []
        for data in self.rgbdata:   
            sample = (data[item])
            RGBsamples.append(sample)
        
        LiDARsamples = []
        for data in self.lidardata:  
            sample = (data[item])
            LiDARsamples.append(sample)
            
            
        return tuple(ysamples),tuple(hsamples),tuple(gpssamples),tuple(RGBsamples), tuple(LiDARsamples)


