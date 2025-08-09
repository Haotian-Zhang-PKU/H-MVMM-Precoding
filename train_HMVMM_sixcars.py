import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import datasets
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import h5py
from model_branch import Multi_Modal_Car1_Model, Multi_Modal_Car2_Model, Multi_Modal_Car3_Model, Multi_Modal_Car4_Model, Multi_Modal_Car5_Model, Multi_Modal_Car6_Model

def load_H_data(SNR_dB, Num_pilots_Time, Car_num):
    data_path = '/dataset/'
    train_car_y_data = []
    train_car_H_data = []
    test_car_y_data = []
    test_car_H_data = []
    car_list = [9,3,1,2,7,10,6]
    for i in range(0, Car_num):
        no = car_list[i]
        car_name = 'car' + str(no)
        print(car_name)
        # Load training data
        train_y_data_path = data_path + 'Train_receive_data_pth/SNR{}/{}' \
                            '/train_Processedreceivesignalv0930_{}_SNR{}pilot{}Fullscenario.pth'.format(SNR_dB, car_name, car_name, SNR_dB, Num_pilots_Time)
        train_H_data_path = data_path + 'Train_DL_H_label_pth/{}' \
                            '/train_DL_Hlabelv0930_{}_SNR{}Fullscenario.pth'.format(car_name, car_name, SNR_dB)

        train_car_y_data.append(torch.load(train_y_data_path))
        train_car_H_data.append(torch.load(train_H_data_path))

        # Load test data
        test_y_data_path = data_path + 'Test_receive_data_pth/SNR{}/{}' \
                           '/test_Processedreceivesignalv0930_{}_SNR{}pilot{}Fullscenario.pth'.format(SNR_dB, car_name, car_name, SNR_dB, Num_pilots_Time)
        test_H_data_path = data_path + 'Test_DL_H_label_pth/{}' \
                           '/test_DL_Hlabelv0930_{}_SNR{}Fullscenario.pth'.format(car_name, car_name, SNR_dB)

        test_car_y_data.append(torch.load(test_y_data_path))
        test_car_H_data.append(torch.load(test_H_data_path))

    return train_car_y_data, train_car_H_data, test_car_y_data, test_car_H_data


def load_GPS_data(Car_num):
    data_path = '/dataset/GPS/'
    train_car_GPS_data = []
    train_car_GPS_data = []
    test_car_GPS_data = []
    test_car_GPS_data = []
    car_list = [9,3,1,2,7,10,6]
    for i in range(0, Car_num):
        no = car_list[i]
        car_name = 'car' + str(no)

        # # Load training data
        train_GPS_path = data_path + 'Train_GPS_pth/{}' \
                            '/train_DL_GPS_{}ve5.pth'.format(car_name, car_name)
        train_car_GPS_data.append(torch.load(train_GPS_path))
        

        # Load test data
        test_GPS_path = data_path + 'Test_GPS_pth/{}' \
                            '/test_DL_GPS_{}ve5.pth'.format(car_name, car_name)
        test_car_GPS_data.append(torch.load(test_GPS_path))
        
        
    return train_car_GPS_data, test_car_GPS_data


def load_RGB_data(Car_num):
    data_path = '/dataset/RGB/'

    train_car_RGB_data = []
    train_car_RGB_data = []
    test_car_RGB_data = []
    test_car_RGB_data = []
    car_list = [9,3,1,2,7,10,6]
    for i in range(0, Car_num):
        no = car_list[i]
        car_name = 'car' + str(no)

        # # Load training data
        train_RGB_path = data_path + 'Train_RGB_pth/{}' \
                            '/train_RGB_{}v2.pth'.format(car_name, car_name)
        train_car_RGB_data.append(torch.load(train_RGB_path))
        

        # Load test data
        test_RGB_path = data_path + 'Test_RGB_pth/{}' \
                            '/test_RGB_{}v2.pth'.format(car_name, car_name)
        test_car_RGB_data.append(torch.load(test_RGB_path))
        
    return train_car_RGB_data, test_car_RGB_data



def load_LiDAR_data(Car_num):
    data_path = '/dataset/LiDAR/'

    train_car_LiDAR_data = []
    test_car_LiDAR_data = []
    car_list = [9,3,1,2,7,10,6]
    for i in range(0, Car_num):
        no = car_list[i]
        car_name = 'car' + str(no)

        # # Load training data
        train_LiDAR_path = data_path + 'Train_LiDAR_pth/{}' \
                            '/train_LiDAR_{}_BEV.pth'.format(car_name, car_name)
        train_car_LiDAR_data.append(torch.load(train_LiDAR_path))
        

        # Load test data
        test_LiDAR_path = data_path + 'Test_LiDAR_pth/{}' \
                            '/test_LiDAR_{}_BEV.pth'.format(car_name, car_name)
        test_car_LiDAR_data.append(torch.load(test_LiDAR_path))
        
    return train_car_LiDAR_data, test_car_LiDAR_data



def PE(p,L):
    p = [t.tolist() for t in p]
    ptheta = np.stack(p[1], axis=-1)
    pphi = np.stack(p[2], axis=-1)
    test = np.stack([ptheta,pphi], axis=1)
    scaled_positions_rho = [x / max(p[0]) for x in p[0]]
    scaled_positions_theta = [x / max(p[1]) for x in p[1]]
    scaled_positions_phi = [x / max(p[2]) for x in p[2]]
    pe_p_rho = np.zeros([2*L, len(scaled_positions_rho)])
    pe_p_theta = np.zeros([2*L, len(scaled_positions_theta)])
    pe_p_phi = np.zeros([2*L, len(scaled_positions_phi)])
    
    # rho
    for j in range(len(scaled_positions_rho)):
        pe_rho_single_batch=[]
        for i in range(L):
            rho = scaled_positions_rho[j]
            freq = 2 ** i
            pe_rho_single_batch.append(np.sin(freq * np.pi * rho))
            pe_rho_single_batch.append(np.cos(freq * np.pi * rho))
        pe_p_rho[:,j] = np.array(pe_rho_single_batch)
        
    # theta
    for j in range(len(scaled_positions_theta)):
        pe_theta_single_batch=[]
        for i in range(L):
            theta = scaled_positions_theta[j]
            freq = 2 ** i
            pe_theta_single_batch.append(np.sin(freq * np.pi * theta))
            pe_theta_single_batch.append(np.cos(freq * np.pi * theta))
        pe_p_theta[:,j] = np.array(pe_theta_single_batch)
        
        
    # phi
    for j in range(len(scaled_positions_phi)):
        pe_phi_single_batch=[]
        for i in range(L):
            phi = scaled_positions_phi[j]
            freq = 2 ** i
            pe_phi_single_batch.append(np.sin(freq * np.pi * phi))
            pe_phi_single_batch.append(np.cos(freq * np.pi * phi))
        pe_p_phi[:,j] = np.array(pe_phi_single_batch)
        
    
    # stack
    rho = np.stack(scaled_positions_rho, axis=-1)
    rho = np.expand_dims(rho,axis=1)
    encoding_rho = np.stack(pe_p_rho, axis=-1)
    encoding_theta = np.stack(pe_p_theta, axis=-1)
    encoding_phi = np.stack(pe_p_phi, axis=-1)
    
    encoding_all = np.concatenate((encoding_theta,encoding_phi), axis=1)
    encoding_all = torch.from_numpy(encoding_all)
    test = torch.from_numpy(test)
    return encoding_all, test

def convert_to_spherical_coordinates(BS_loc, car_loc):

    dx = car_loc[:,0] - BS_loc[0]
    dy = car_loc[:,1] - BS_loc[1]
    dz = car_loc[:,2] - BS_loc[2]

    # coordinates to spherical coordinates
    rho = np.sqrt(dx**2 + dy**2 + dz**2)  
    theta = np.arctan2(dy, dx)  
    phi = np.arccos(dz / rho)  

    return rho, theta, phi


   

class SE_loss_w_threshold(nn.Module):
    def __init__(self):
        super().__init__()
        # self.h = h
        # self.v = v
        # self.N = N
        
    def forward(self, h, v, N, device):
        num_user, batch_size, num_Nt = h.size()
        Rp = torch.zeros(batch_size)
        Rp = Rp.to(device)
        v_norm = torch.zeros(num_user, batch_size, int(num_Nt/2), dtype=torch.complex128).to(device)
        R = 0
        T = 0.3
        R_per_user=[]
        R_per_user_p=[]

        for j in range(batch_size):
            v_sample = v[:,j,:]
            v_sample = v_sample[:,0:int(num_Nt/2)] + v_sample[:,int(num_Nt/2):num_Nt]*1j
            current_v_norm = torch.norm(v_sample, p='fro')
            scale = np.sqrt(num_user) / current_v_norm
            v_sample_scale = v_sample * scale
            v_norm[:,j,:] = v_sample_scale
            
            
        for i in range(num_user): 
           h_user = h[i,:]
           h_user = h_user[:,0:int(num_Nt/2)] + h_user[:,int(num_Nt/2):num_Nt]*1j
           v_user = v_norm[i,:]
           v_user_H = v_user.conj().T
           S_per_sample = torch.diag(torch.matmul(h_user, v_user_H))  
           Sp = torch.square(torch.abs(S_per_sample))  
           Ip = torch.zeros_like(Sp) 
           for j in range(num_user):
              if j != i:
                 v_inter = v[j,:]
                 v_inter = v_inter[:,0:int(num_Nt/2)] + v_inter[:,int(num_Nt/2):num_Nt]*1j 
                 v_inter_H = v_inter.conj().T
                 I_per_sample = torch.diag(torch.matmul(h_user, v_inter_H))
                 Ip = Ip + torch.square(torch.abs(I_per_sample))
            
           SINRp = torch.div(Sp,Ip+N)      
           R_userp = torch.log2(torch.add(1,SINRp))
           R_per_user_p.append(torch.mean(R_userp))

           Rp = torch.sub(Rp,R_userp)
           diff = [torch.sub(T,x) for x in R_userp]
           penalty  = [torch.pow(10,x) for x in diff] 
           Rp = torch.add(Rp,torch.stack(penalty))
        R_realp = torch.sum(torch.stack(R_per_user_p))
        return  torch.mean(Rp), R_per_user_p, R_realp


def quantize_complex_tensor(input_tensor, quant_bits, quant_min, quant_max):
    step_size = (quant_max - quant_min) / (2**quant_bits)
    
    # Quantize the real and imaginary parts
    quantized = torch.floor((input_tensor - quant_min) / step_size) * step_size + quant_min
    quantized = torch.clamp(quantized, min=quant_min, max=quant_max)  # Truncate values outside the range

    return quantized

            
def MU_get_data_trainloaders(train_batch_size, train_H_data, train_y_data, train_gps_data, train_rgb_data, train_lidar_data,num_pilot):
    train_data = datasets.Dataset_3MMFF(train_y_data, train_H_data, train_gps_data, train_rgb_data, train_lidar_data,num_pilot)
    MU_train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    return MU_train_loader

def MU_get_data_testloaders(test_batch_size, test_H_data, test_y_data, test_gps_data, test_rgb_data, test_lidar_data,num_pilot):
    test_data = datasets.Dataset_3MMFF(test_y_data, test_H_data, test_gps_data, test_rgb_data, test_lidar_data,num_pilot)
    MU_test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)
    return MU_test_loader



def vehicle_sensors_description(input_vector):
    sensors = ["pilot", "GPS", "RGB", "LiDAR"]
    description = []

    for i in range(len(input_vector)):
        if input_vector[i] == 1:
            description.append("equipped with" + sensors[i])
        else:
            description.append("without" + sensors[i])

    return ",".join(description)



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

def main():
    Average=[]
    Average_q=[]

    for time in range(1):
        Nt = 128
        Num_user = 6
        Power = 6
        Num_epochs = 400
        SNR_dB = 21
        Num_pilots_Time = 8
        Pilot_power = 1e-1
        BATCH_size = 32
        quan_bit = 4
        quan_max = 0.002
        quan_min = -0.002
        Signal_size = 2 * Num_pilots_Time 
        Signal_feature_length = 128
        L= 5
        GPS_size = L*2*2
        # GPS_size = 2
        GPS_feature_length = 256
        RGB_size = 36
        RGB_feature_length = 256
        LiDAR_size = 64
        LiDAR_feature_length = 512
        # load data of multiple cars
        train_car_y_data, train_car_H_data, test_car_y_data, test_car_H_data = load_H_data(SNR_dB, 64, Num_user)
        train_car_GPS_data, test_car_GPS_data = load_GPS_data(Num_user)
        train_car_RGB_data, test_car_RGB_data = load_RGB_data(Num_user)
        train_car_LiDAR_data, test_car_LiDAR_data = load_LiDAR_data(Num_user)
        file_name = "/data/GPS_BS.mat"
        data = h5py.File(file_name)
        BS_loc = data['GPS_BS'][:] 
    
        train_per_usr_rate = []  
        val_per_usr_rate = []  
        val_per_usr_rate_q = [] 
        for _ in range(Num_user):
            train_new_list = [] 
            train_per_usr_rate.append(train_new_list) 
            val_new_list = [] 
            val_per_usr_rate.append(val_new_list)  
            val_new_list_q = []  
            val_per_usr_rate_q.append(val_new_list_q)  
            
            
        # Train
        train_loader = MU_get_data_trainloaders(BATCH_size, train_car_H_data, train_car_y_data, train_car_GPS_data, train_car_RGB_data, train_car_LiDAR_data, Num_pilots_Time)

        # Test
        test_loader = MU_get_data_testloaders(399, test_car_H_data, test_car_y_data, test_car_GPS_data, test_car_RGB_data, test_car_LiDAR_data, Num_pilots_Time)


        local_models = []
        indicator=[]
        select_car1 =  [1,1,0,1]   # Available modality indicator vector for car1 
        indicator.append(select_car1)
        local_models.append(Multi_Modal_Car1_Model(Signal_size,GPS_size, RGB_size, LiDAR_size, Signal_feature_length, GPS_feature_length, RGB_feature_length, LiDAR_feature_length, Power, select_car1, Num_pilots_Time).double().to(device))
        select_car2 =  [1,1,0,0]
        indicator.append(select_car2)
        local_models.append(Multi_Modal_Car2_Model(Signal_size,GPS_size, RGB_size, LiDAR_size, Signal_feature_length, GPS_feature_length, RGB_feature_length, LiDAR_feature_length, Power, select_car2, Num_pilots_Time).double().to(device))
        select_car3 =  [1,0,1,0]
        indicator.append(select_car3)
        local_models.append(Multi_Modal_Car3_Model(Signal_size,GPS_size, RGB_size, LiDAR_size, Signal_feature_length, GPS_feature_length, RGB_feature_length, LiDAR_feature_length, Power, select_car3, Num_pilots_Time).double().to(device))
        select_car4 =  [1,0,0,1]
        indicator.append(select_car4)
        local_models.append(Multi_Modal_Car4_Model(Signal_size,GPS_size, RGB_size, LiDAR_size, Signal_feature_length, GPS_feature_length, RGB_feature_length, LiDAR_feature_length, Power, select_car4, Num_pilots_Time).double().to(device))
        select_car5 =  [1,1,1,0]
        indicator.append(select_car5)
        local_models.append(Multi_Modal_Car5_Model(Signal_size,GPS_size, RGB_size, LiDAR_size, Signal_feature_length, GPS_feature_length, RGB_feature_length, LiDAR_feature_length, Power, select_car5, Num_pilots_Time).double().to(device))
        select_car6 =  [1,1,0,1]
        indicator.append(select_car6)
        local_models.append(Multi_Modal_Car6_Model(Signal_size,GPS_size, RGB_size, LiDAR_size, Signal_feature_length, GPS_feature_length, RGB_feature_length, LiDAR_feature_length, Power, select_car6, Num_pilots_Time).double().to(device))

        
        optimizers = [optim.Adam(model.parameters(), lr=0.001/10) for model in local_models]
        loss_func = SE_loss_w_threshold()
        LR_sch_list = []
        count = 0
        for opt in optimizers:    
            LR_sch = torch.optim.lr_scheduler.MultiStepLR(opt, [100,150], gamma=1, last_epoch=-1) 
            LR_sch_list.append(LR_sch)

        # ensure the directory to save the models exists
        models_dir = "/data2/Haotiandata/FDD/Model"
        os.makedirs(models_dir, exist_ok=True)
        loss_sum = []
        val_sum = []
        actual_val_sum=[]
        step_sum = []

        for epoch in range(Num_epochs):
            print(f"Epoch {epoch+1}")
            print(f"Num of Pilots {Num_pilots_Time}")
            with tqdm(total=len(train_loader), desc="Training", unit="batch") as pbar:
                total_loss = 0
                total_real_sumrate = 0
                cnt_train = 0
                ave_per_user_loss = np.zeros(6)
                total_per_user_loss = np.zeros(6)
                for batch, (y,H,gps,rgb,lidar) in enumerate(train_loader):
                    y_elements = y
                    H_elements = H
                    gps_elements = gps
                    rgb_elements = rgb
                    lidar_elements = lidar

                    # Apply device transformation to each element in y
                    y_transformed = []
                    for y_element in y_elements:
                        y_transformed.append(y_element.to(device))

                    # Apply device transformation to each element in H
                    H_transformed = []
                    for H_element in H_elements:
                        H_transformed.append(H_element.to(device))

                    # Apply device transformation to each element in gps
                    gps_transformed = []
                    for gps_element in gps_elements:
                        gps_transformed.append(gps_element)
                    
                    # Apply device transformation to each element in rgb
                    rgb_transformed = []
                    for rgb_element in rgb_elements:
                        rgb_transformed.append(rgb_element)
                    
                    lidar_transformed = []
                    for lidar_element in lidar_elements:
                        lidar_transformed.append(lidar_element)
                        
                        
                    local_outputs_car1 = []
                    local_outputs_car2 = []
                    local_outputs_car3 = []
                    local_outputs_car4 = []
                    local_outputs_car5 = []
                    local_outputs_car6 = []

                    DL_H_car1 = []
                    DL_H_car2 = []
                    DL_H_car3 = []
                    DL_H_car4 = []
                    DL_H_car5 = []
                    DL_H_car6 = []

                    for i, model in enumerate(local_models):   
                        model.train()
                        if i == 0:   
                            optimizer_1 = optimizers[i]
                            optimizer_1.zero_grad()
                            y1 = y_transformed[i]
                            H1 = H_transformed[i]
                            car1_gps = gps_transformed[i]
                            car1_rgb = rgb_transformed[i]
                            car1_lidar = lidar_transformed[i]
                            car1_lidar = car1_lidar.double()
                            sph_gps1 = convert_to_spherical_coordinates(BS_loc, car1_gps)
                            PE_sph_gps1,test = PE(sph_gps1,L) 
                            PE_sph_gps1 = PE_sph_gps1.to(device)
                            car1_rgb = car1_rgb.to(device)
                            car1_lidar = car1_lidar.to(device)
                            output_car1 = model(y1,PE_sph_gps1,car1_rgb,car1_lidar,select_car1)
                            local_outputs_car1.append( output_car1)
                            DL_H_car1.append(H1)
                        elif i == 1:
                            optimizer_2 = optimizers[i]
                            optimizer_2.zero_grad()
                            y2 = y_transformed[i]
                            H2 = H_transformed[i]
                            car2_gps = gps_transformed[i]
                            car2_rgb = rgb_transformed[i]
                            car2_lidar = lidar_transformed[i]
                            car2_lidar = car2_lidar.double()
                            sph_gps2 = convert_to_spherical_coordinates(BS_loc, car2_gps)
                            PE_sph_gps2,test = PE(sph_gps2,L)  
                            PE_sph_gps2 = PE_sph_gps2.to(device)
                            car2_rgb = car2_rgb.to(device)
                            car2_lidar = car2_lidar.to(device)
                            output_car2 = model(y2,PE_sph_gps2,car2_rgb,car2_lidar,select_car2)
                            local_outputs_car2.append( output_car2)

                            DL_H_car2.append(H2)
                        elif i == 2:
                            optimizer_3 = optimizers[i]
                            optimizer_3.zero_grad()
                            y3 = y_transformed[i]
                            H3 = H_transformed[i]
                            car3_gps = gps_transformed[i]
                            car3_rgb = rgb_transformed[i]
                            car3_lidar = lidar_transformed[i]
                            car3_lidar = car3_lidar.double()
                            sph_gps3 = convert_to_spherical_coordinates(BS_loc, car3_gps)
                            PE_sph_gps3,test = PE(sph_gps3,L)  
                            PE_sph_gps3 = PE_sph_gps3.to(device)
                            car3_rgb = car3_rgb.to(device)
                            car3_lidar = car3_lidar.to(device)
                            output_car3 = model(y3,PE_sph_gps3,car3_rgb,car3_lidar,select_car3)
                            local_outputs_car3.append( output_car3)
                            DL_H_car3.append(H3)
                        elif i == 3:
                            optimizer_4 = optimizers[i]
                            optimizer_4.zero_grad()
                            y4 = y_transformed[i]
                            H4 = H_transformed[i]
                            car4_gps = gps_transformed[i]
                            car4_rgb = rgb_transformed[i]
                            car4_lidar = lidar_transformed[i]
                            car4_lidar = car4_lidar.double()
                            sph_gps4 = convert_to_spherical_coordinates(BS_loc, car4_gps)
                            PE_sph_gps4,test = PE(sph_gps4,L)  
                            PE_sph_gps4 = PE_sph_gps4.to(device)
                            car4_rgb = car4_rgb.to(device)
                            car4_lidar = car4_lidar.to(device)
                            output_car4 = model(y4,PE_sph_gps4,car4_rgb,car4_lidar,select_car4)
                            local_outputs_car4.append( output_car4)
                            DL_H_car4.append(H4)
                        elif i == 4:
                            optimizer_5 = optimizers[i]
                            optimizer_5.zero_grad()
                            y5 = y_transformed[i]
                            H5 = H_transformed[i]
                            car5_gps = gps_transformed[i]
                            car5_rgb = rgb_transformed[i]
                            car5_lidar = lidar_transformed[i]
                            car5_lidar = car5_lidar.double()
                            sph_gps5 = convert_to_spherical_coordinates(BS_loc, car5_gps)
                            PE_sph_gps5,test = PE(sph_gps5,L)  
                            PE_sph_gps5 = PE_sph_gps5.to(device)
                            car5_rgb = car5_rgb.to(device)
                            car5_lidar = car5_lidar.to(device)
                            output_car5 = model(y5,PE_sph_gps5,car5_rgb,car5_lidar,select_car5)
                            local_outputs_car5.append( output_car5)
                            DL_H_car5.append(H5)
                        elif i == 5:
                            optimizer_6 = optimizers[i]
                            optimizer_6.zero_grad()
                            y6 = y_transformed[i]
                            H6 = H_transformed[i]
                            car6_gps = gps_transformed[i]
                            car6_rgb = rgb_transformed[i]
                            car6_lidar = lidar_transformed[i]
                            car6_lidar = car6_lidar.double()
                            sph_gps6 = convert_to_spherical_coordinates(BS_loc, car6_gps)
                            PE_sph_gps6,test = PE(sph_gps6,L)  
                            PE_sph_gps6 = PE_sph_gps6.to(device)
                            car6_rgb = car6_rgb.to(device)
                            car6_lidar = car6_lidar.to(device)
                            output_car6 = model(y6,PE_sph_gps6,car6_rgb,car6_lidar,select_car6)
                            local_outputs_car6.append(output_car6)
                            DL_H_car6.append(H6)

                        

                    # Aggregate the outputs from the local models
                    local_outputs_car1 = torch.stack(local_outputs_car1)
                    local_outputs_car2 = torch.stack(local_outputs_car2)
                    local_outputs_car3 = torch.stack(local_outputs_car3)
                    local_outputs_car4 = torch.stack(local_outputs_car4)
                    local_outputs_car5 = torch.stack(local_outputs_car5)
                    local_outputs_car6 = torch.stack(local_outputs_car6)

                    aggregated_precoder_output = torch.stack((local_outputs_car1,local_outputs_car2,local_outputs_car3,local_outputs_car4,local_outputs_car5,local_outputs_car6), dim=1)
                    aggregated_precoder_output = torch.squeeze(aggregated_precoder_output)
            
                    DL_H_car1 = torch.stack(DL_H_car1)
                    DL_H_car2 = torch.stack(DL_H_car2)
                    DL_H_car3 = torch.stack(DL_H_car3)
                    DL_H_car4 = torch.stack(DL_H_car4)
                    DL_H_car5 = torch.stack(DL_H_car5)
                    DL_H_car6 = torch.stack(DL_H_car6)

                    DL_MU_H_label =  torch.stack((DL_H_car1 , DL_H_car2, DL_H_car3, DL_H_car4, DL_H_car5, DL_H_car6), dim=1)
                    DL_MU_H_label = torch.squeeze(DL_MU_H_label)
                    SNR_ = 10**(SNR_dB / 10)
                    
                   
                    noise_power = Pilot_power / SNR_   

                    global_loss, global_loss_per_user, real_sumrate = loss_func(DL_MU_H_label, aggregated_precoder_output, noise_power, device)   

                
                    # Backward pass
                    global_loss.backward()
                
                    # Update each local model
                    for i, model in enumerate(local_models):
                        optimizer = optimizers[i]
                        optimizer.step()
                    cnt_train += 1
                    total_loss += global_loss.item()
                    total_real_sumrate += real_sumrate.item()
                    for index in range(Num_user):
                        total_per_user_loss[index] +=  global_loss_per_user[index].item()
                        ave_per_user_loss[index] = np.round(total_per_user_loss[index]/ cnt_train * 1000) / 1000 
                    # Update the progress bar with the latest loss values
                    pbar.update(1)
                    pbar.set_postfix_str(f"Global Loss: {global_loss.item():.4f}, A_Global Loss: {total_loss/cnt_train:.4f}, Per_user: {[ave_per_user_loss[i] for i in range(6)]}, Real sumrate: {real_sumrate:.4f}")
            
                for i in range(Num_user):
                    LR_sch = LR_sch_list[i]
                    LR_sch.step()
                    train_per_usr_rate[i].append(ave_per_user_loss[i])
                loss_sum.append(total_real_sumrate/cnt_train)
                step_sum.append(cnt_train)
                
                with torch.no_grad():
                    Val_loss = 0
                    Val_loss_q = 0
                    Val_real_sumrate = 0
                    Val_real_sumrate_q = 0
                    cnt_val = 0
                    val_ave_per_user_rate = np.zeros(6)
                    val_total_per_user_rate = np.zeros(6)
                    val_ave_per_user_rate_q = np.zeros(6)
                    val_total_per_user_rate_q = np.zeros(6)

                    for batch, (y,H,gps,rgb,lidar) in enumerate(test_loader):
                            # Unpack elements from tuples
                            y_elements = y
                            H_elements = H
                            gps_elements = gps
                            rgb_elements = rgb
                            lidar_elements = lidar

                            # Apply device transformation to each element in y
                            y_transformed = []
                            for y_element in y_elements:
                                y_transformed.append(y_element.to(device))

                            # Apply device transformation to each element in H
                            H_transformed = []
                            for H_element in H_elements:
                                H_transformed.append(H_element.to(device))

                            # Apply device transformation to each element in gps
                            gps_transformed = []
                            for gps_element in gps_elements:
                                gps_transformed.append(gps_element)
                            
                            # Apply device transformation to each element in rgb
                            rgb_transformed = []
                            for rgb_element in rgb_elements:
                                rgb_transformed.append(rgb_element)
                                
                            lidar_transformed = []
                            for lidar_element in lidar_elements:
                                lidar_transformed.append(lidar_element)
                        
                            local_outputs_car1 = []
                            local_outputs_car2 = []
                            local_outputs_car3 = []
                            local_outputs_car4 = []
                            local_outputs_car5 = []
                            local_outputs_car6 = []

                            DL_H_car1 = []
                            DL_H_car2 = []
                            DL_H_car3 = []
                            DL_H_car4 = []
                            DL_H_car5 = []
                            DL_H_car6 = []
                    
                            for i, model in enumerate(local_models):  
                                model.eval()
                                if i == 0:   
                                    optimizer_1 = optimizers[i]
                                    optimizer_1.zero_grad()
                                    y1 = y_transformed[i]
                                    H1 = H_transformed[i]
                                    car1_gps = gps_transformed[i]
                                    car1_rgb = rgb_transformed[i]
                                    car1_lidar = lidar_transformed[i]
                                    car1_lidar = car1_lidar.double()
                                    sph_gps1 = convert_to_spherical_coordinates(BS_loc, car1_gps)
                                    PE_sph_gps1,test = PE(sph_gps1,L)  
                                    PE_sph_gps1 = PE_sph_gps1.to(device)
                                    car1_rgb = car1_rgb.to(device)
                                    car1_lidar = car1_lidar.to(device)
                                    output_car1 = model(y1,PE_sph_gps1,car1_rgb,car1_lidar,select_car1)
                                    local_outputs_car1.append( output_car1)
                                    DL_H_car1.append(H1)
                                elif i == 1:
                                    optimizer_2 = optimizers[i]
                                    optimizer_2.zero_grad()
                                    y2 = y_transformed[i]
                                    H2 = H_transformed[i]
                                    car2_gps = gps_transformed[i]
                                    car2_rgb = rgb_transformed[i]
                                    car2_lidar = lidar_transformed[i]
                                    car2_lidar = car2_lidar.double()
                                    sph_gps2 = convert_to_spherical_coordinates(BS_loc, car2_gps)
                                    PE_sph_gps2,test = PE(sph_gps2,L)  
                                    PE_sph_gps2 = PE_sph_gps2.to(device)
                                    car2_rgb = car2_rgb.to(device)
                                    car2_lidar = car2_lidar.to(device)
                                    output_car2 = model(y2,PE_sph_gps2,car2_rgb,car2_lidar,select_car2)
                                    local_outputs_car2.append( output_car2)
                                    DL_H_car2.append(H2)
                                elif i == 2:
                                    optimizer_3 = optimizers[i]
                                    optimizer_3.zero_grad()
                                    y3 = y_transformed[i]
                                    H3 = H_transformed[i]
                                    car3_gps = gps_transformed[i]
                                    car3_rgb = rgb_transformed[i]
                                    car3_lidar = lidar_transformed[i]
                                    car3_lidar = car3_lidar.double()
                                    sph_gps3 = convert_to_spherical_coordinates(BS_loc, car3_gps)
                                    PE_sph_gps3,test = PE(sph_gps3,L)  
                                    PE_sph_gps3 = PE_sph_gps3.to(device)
                                    car3_rgb = car3_rgb.to(device)
                                    car3_lidar = car3_lidar.to(device)
                                    output_car3 = model(y3,PE_sph_gps3,car3_rgb,car3_lidar,select_car3)
                                    local_outputs_car3.append( output_car3)
                                    DL_H_car3.append(H3)
                                elif i == 3:
                                    optimizer_4 = optimizers[i]
                                    optimizer_4.zero_grad()
                                    y4 = y_transformed[i]
                                    H4 = H_transformed[i]
                                    car4_gps = gps_transformed[i]
                                    car4_rgb = rgb_transformed[i]
                                    car4_lidar = lidar_transformed[i]
                                    car4_lidar = car4_lidar.double()
                                    sph_gps4 = convert_to_spherical_coordinates(BS_loc, car4_gps)
                                    PE_sph_gps4,test = PE(sph_gps4,L)  
                                    PE_sph_gps4 = PE_sph_gps4.to(device)
                                    car4_rgb = car4_rgb.to(device)
                                    car4_lidar = car4_lidar.to(device)
                                    output_car4 = model(y4,PE_sph_gps4,car4_rgb,car4_lidar,select_car4)
                                    local_outputs_car4.append( output_car4)
                                    DL_H_car4.append(H4)
                                elif i == 4:
                                    y5 = y_transformed[i]
                                    H5 = H_transformed[i]
                                    car5_gps = gps_transformed[i]
                                    car5_rgb = rgb_transformed[i]
                                    car5_lidar = lidar_transformed[i]
                                    car5_lidar = car5_lidar.double()
                                    sph_gps5 = convert_to_spherical_coordinates(BS_loc, car5_gps)
                                    PE_sph_gps5,test = PE(sph_gps5,L)  
                                    PE_sph_gps5 = PE_sph_gps5.to(device)
                                    car5_rgb = car5_rgb.to(device) 
                                    car5_lidar = car5_lidar.to(device)
                                    output_car5 = model(y5,PE_sph_gps5,car5_rgb ,car5_lidar,select_car5)
                                    local_outputs_car5.append( output_car5)
                                    DL_H_car5.append(H5)
                                elif i == 5:
                                    y6 = y_transformed[i]
                                    H6 = H_transformed[i]
                                    car6_gps = gps_transformed[i]
                                    car6_rgb = rgb_transformed[i]
                                    car6_lidar = lidar_transformed[i]
                                    car6_lidar = car6_lidar.double()
                                    sph_gps6 = convert_to_spherical_coordinates(BS_loc, car6_gps)
                                    PE_sph_gps6,test = PE(sph_gps6,L)  
                                    PE_sph_gps6 = PE_sph_gps6.to(device)
                                    car6_rgb = car6_rgb.to(device) 
                                    car6_lidar = car6_lidar.to(device)
                                    output_car6 = model(y6,PE_sph_gps6,car6_rgb ,car6_lidar,select_car6)
                                    local_outputs_car6.append( output_car6)
                                    DL_H_car6.append(H6)
                                            
                                

                            # Aggregate the outputs from the local models
                            local_outputs_car1 = torch.stack(local_outputs_car1)
                            local_outputs_car2 = torch.stack(local_outputs_car2)
                            local_outputs_car3 = torch.stack(local_outputs_car3)
                            local_outputs_car4 = torch.stack(local_outputs_car4)
                            local_outputs_car5 = torch.stack(local_outputs_car5)
                            local_outputs_car6 = torch.stack(local_outputs_car6)

                            aggregated_precoder_output = torch.stack((local_outputs_car1,local_outputs_car2,local_outputs_car3,local_outputs_car4,local_outputs_car5,local_outputs_car6), dim=1)
                            aggregated_precoder_output = torch.squeeze(aggregated_precoder_output)
                            aggregated_precoder_output_q = quantize_complex_tensor(aggregated_precoder_output, 2, quan_min, quan_max)

                            DL_H_car1 = torch.stack(DL_H_car1)
                            DL_H_car2 = torch.stack(DL_H_car2)
                            DL_H_car3 = torch.stack(DL_H_car3)
                            DL_H_car4 = torch.stack(DL_H_car4)
                            DL_H_car5 = torch.stack(DL_H_car5)
                            DL_H_car6 = torch.stack(DL_H_car6)

                            DL_MU_H_label =  torch.stack((DL_H_car1 , DL_H_car2, DL_H_car3, DL_H_car4, DL_H_car5, DL_H_car6), dim=1)
                            DL_MU_H_label = torch.squeeze(DL_MU_H_label)
                            H = DL_MU_H_label.cpu().numpy()
                            
                            SNR_ = 10**(SNR_dB / 10)
                            noise_power = Pilot_power / SNR_   
                            val_global_loss, global_loss_per_user, val_real_sum_rate = loss_func(DL_MU_H_label, aggregated_precoder_output, noise_power, device)   
                            val_global_loss_q, global_loss_per_user_q, val_real_sum_rate_q = loss_func(DL_MU_H_label, aggregated_precoder_output_q, noise_power, device)   
                            cnt_val += 1
                            Val_loss += val_global_loss.item()
                            Val_real_sumrate += val_real_sum_rate.item()
                            Val_loss_q += val_global_loss_q.item()
                            Val_real_sumrate_q += val_real_sum_rate_q.item()


                            for i in range(Num_user):
                                val_total_per_user_rate[i] +=  global_loss_per_user[i].item()
                                val_total_per_user_rate_q[i] +=  global_loss_per_user_q[i].item()
                                val_ave_per_user_rate[i] = np.round(val_total_per_user_rate[i]/ cnt_val * 1000) / 1000
                                val_ave_per_user_rate_q[i] = np.round(val_total_per_user_rate_q[i]/ cnt_val * 1000) / 1000
                    #per batch
                    print(f'MMMTrain_loss: {total_loss/cnt_train:.4f}')
                    print(f'Val_loss: {Val_loss/cnt_val:.4f}')
                    print(f'Sum rate (wo quant): {Val_real_sumrate/cnt_val:.4f}')
                    print(f'Sum rate (quant): {Val_real_sumrate_q/cnt_val:.4f}')
                    for i in range(Num_user):
                        val_per_usr_rate[i].append(val_ave_per_user_rate[i])
                        val_per_usr_rate_q[i].append(val_ave_per_user_rate_q[i])
                    print(f'User rate (wo quant): {val_per_usr_rate[0][-1]:.4f},{val_per_usr_rate[1][-1].item():.4f},{val_per_usr_rate[2][-1].item():.4f},{val_per_usr_rate[3][-1].item():.4f},{val_per_usr_rate[4][-1].item():.4f},{val_per_usr_rate[5][-1].item():.4f}')
                    print(f'User rate (quant): {val_per_usr_rate_q[0][-1]:.4f},{val_per_usr_rate_q[1][-1].item():.4f},{val_per_usr_rate_q[2][-1].item():.4f},{val_per_usr_rate_q[3][-1].item():.4f},{val_per_usr_rate_q[4][-1].item():.4f},{val_per_usr_rate_q[5][-1].item():.4f}')
                    val_sum.append(Val_real_sumrate/cnt_val)
                    actual_val_sum.append(Val_real_sumrate_q/cnt_val)

                    
                        

        with torch.no_grad():
            Val_loss = 0
            cnt_val = 0
            Val_real_sumrate = 0
            Val_loss_q = 0
            Val_real_sumrate_q = 0

            
            for batch, (y,H,gps,rgb,lidar) in enumerate(test_loader):
            # Unpack elements from tuples
                y_elements = y
                H_elements = H
                gps_elements = gps
                rgb_elements = rgb
                lidar_elements = lidar

                # Apply device transformation to each element in y
                y_transformed = []
                for y_element in y_elements:
                    y_transformed.append(y_element.to(device))

                # Apply device transformation to each element in H
                H_transformed = []
                for H_element in H_elements:
                    H_transformed.append(H_element.to(device))

                # Apply device transformation to each element in gps
                gps_transformed = []
                for gps_element in gps_elements:
                    gps_transformed.append(gps_element)
                    
                # Apply device transformation to each element in rgb
                rgb_transformed = []
                for rgb_element in rgb_elements:
                    rgb_transformed.append(rgb_element)
                    
                lidar_transformed = []
                for lidar_element in lidar_elements:
                    lidar_transformed.append(lidar_element)
                
                local_outputs_car1 = []
                local_outputs_car2 = []
                local_outputs_car3 = []
                local_outputs_car4 = []
                local_outputs_car5 = []
                local_outputs_car6 = []

                DL_H_car1 = []
                DL_H_car2 = []
                DL_H_car3 = []
                DL_H_car4 = []
                DL_H_car5 = []
                DL_H_car6 = []
        
                for i, model in enumerate(local_models):   
                    model.eval()
                    if i == 0:   
                        optimizer_1 = optimizers[i]
                        optimizer_1.zero_grad()
                        y1 = y_transformed[i]
                        H1 = H_transformed[i]
                        car1_gps = gps_transformed[i]
                        car1_rgb = rgb_transformed[i]
                        car1_lidar = lidar_transformed[i]
                        car1_lidar = car1_lidar.double()
                        sph_gps1 = convert_to_spherical_coordinates(BS_loc, car1_gps)
                        PE_sph_gps1,test = PE(sph_gps1,L)  
                        PE_sph_gps1 = PE_sph_gps1.to(device)
                        car1_rgb = car1_rgb.to(device)
                        car1_lidar = car1_lidar.to(device)
                        output_car1 = model(y1,PE_sph_gps1,car1_rgb,car1_lidar,select_car1)
                        local_outputs_car1.append( output_car1)
                        DL_H_car1.append(H1)
                    elif i == 1:
                        optimizer_2 = optimizers[i]
                        optimizer_2.zero_grad()
                        y2 = y_transformed[i]
                        H2 = H_transformed[i]
                        car2_gps = gps_transformed[i]
                        car2_rgb = rgb_transformed[i]
                        car2_lidar = lidar_transformed[i]
                        car2_lidar = car2_lidar.double()
                        sph_gps2 = convert_to_spherical_coordinates(BS_loc, car2_gps)
                        PE_sph_gps2,test = PE(sph_gps2,L)  
                        PE_sph_gps2 = PE_sph_gps2.to(device)
                        car2_rgb = car2_rgb.to(device)
                        car2_lidar = car2_lidar.to(device)
                        output_car2 = model(y2,PE_sph_gps2,car2_rgb,car2_lidar,select_car2)
                        local_outputs_car2.append( output_car2)
                        DL_H_car2.append(H2)
                    elif i == 2:
                        optimizer_3 = optimizers[i]
                        optimizer_3.zero_grad()
                        y3 = y_transformed[i]
                        H3 = H_transformed[i]
                        car3_gps = gps_transformed[i]
                        car3_rgb = rgb_transformed[i]
                        car3_lidar = lidar_transformed[i]
                        car3_lidar = car3_lidar.double()
                        sph_gps3 = convert_to_spherical_coordinates(BS_loc, car3_gps)
                        PE_sph_gps3,test = PE(sph_gps3,L)  
                        PE_sph_gps3 = PE_sph_gps3.to(device)
                        car3_rgb = car3_rgb.to(device)
                        car3_lidar = car3_lidar.to(device)
                        output_car3 = model(y3,PE_sph_gps3,car3_rgb,car3_lidar,select_car3)
                        local_outputs_car3.append( output_car3)
                        DL_H_car3.append(H3)
                    elif i == 3:
                        optimizer_4 = optimizers[i]
                        optimizer_4.zero_grad()
                        y4 = y_transformed[i]
                        H4 = H_transformed[i]
                        car4_gps = gps_transformed[i]
                        car4_rgb = rgb_transformed[i]
                        car4_lidar = lidar_transformed[i]
                        car4_lidar = car4_lidar.double()
                        sph_gps4 = convert_to_spherical_coordinates(BS_loc, car4_gps)
                        PE_sph_gps4,test = PE(sph_gps4,L)  
                        PE_sph_gps4 = PE_sph_gps4.to(device)
                        car4_rgb = car4_rgb.to(device)
                        car4_lidar = car4_lidar.to(device)
                        output_car4 = model(y4,PE_sph_gps4,car4_rgb,car4_lidar,select_car4)
                        local_outputs_car4.append( output_car4)
                        DL_H_car4.append(H4)
                    elif i == 4:
                        y5 = y_transformed[i]
                        H5 = H_transformed[i]
                        car5_gps = gps_transformed[i]
                        car5_rgb = rgb_transformed[i]
                        car5_lidar = lidar_transformed[i]
                        car5_lidar = car5_lidar.double()
                        sph_gps5 = convert_to_spherical_coordinates(BS_loc, car5_gps)
                        PE_sph_gps5,test = PE(sph_gps5,L)  
                        PE_sph_gps5 = PE_sph_gps5.to(device)
                        car5_rgb = car5_rgb.to(device) 
                        car5_lidar = car5_lidar.to(device)
                        output_car5 = model(y5,PE_sph_gps5,car5_rgb ,car5_lidar,select_car5)
                        local_outputs_car5.append( output_car5)
                        DL_H_car5.append(H5)
                    elif i == 5:
                        y6 = y_transformed[i]
                        H6 = H_transformed[i]
                        car6_gps = gps_transformed[i]
                        car6_rgb = rgb_transformed[i]
                        car6_lidar = lidar_transformed[i]
                        car6_lidar = car6_lidar.double()
                        sph_gps6 = convert_to_spherical_coordinates(BS_loc, car6_gps)
                        PE_sph_gps6,test = PE(sph_gps6,L)  
                        PE_sph_gps6 = PE_sph_gps6.to(device)
                        car6_rgb = car6_rgb.to(device) 
                        car6_lidar = car6_lidar.to(device)
                        output_car6 = model(y6,PE_sph_gps6,car6_rgb ,car6_lidar,select_car6)
                        local_outputs_car6.append( output_car6)
                        DL_H_car6.append(H6)
                        

                # Aggregate the outputs from the local models
                local_outputs_car1 = torch.stack(local_outputs_car1)
                local_outputs_car2 = torch.stack(local_outputs_car2)
                local_outputs_car3 = torch.stack(local_outputs_car3)
                local_outputs_car4 = torch.stack(local_outputs_car4)
                local_outputs_car5 = torch.stack(local_outputs_car5)
                local_outputs_car6 = torch.stack(local_outputs_car6)

                aggregated_precoder_output = torch.stack((local_outputs_car1,local_outputs_car2,local_outputs_car3,local_outputs_car4,local_outputs_car5,local_outputs_car6), dim=1)
                aggregated_precoder_output = torch.squeeze(aggregated_precoder_output)
                
                max_value = torch.max(aggregated_precoder_output).item()
                min_value = torch.min(aggregated_precoder_output).item()
                max_value = 0.002
                min_value = -0.002
                aggregated_precoder_output_q = quantize_complex_tensor(aggregated_precoder_output, quan_bit, min_value, max_value)
                ##############################
                precoder = aggregated_precoder_output.cpu().numpy()

                DL_H_car1 = torch.stack(DL_H_car1)
                DL_H_car2 = torch.stack(DL_H_car2)
                DL_H_car3 = torch.stack(DL_H_car3)
                DL_H_car4 = torch.stack(DL_H_car4)
                DL_H_car5 = torch.stack(DL_H_car5)
                DL_H_car6 = torch.stack(DL_H_car6)

                DL_MU_H_label =  torch.stack((DL_H_car1 , DL_H_car2, DL_H_car3, DL_H_car4, DL_H_car5, DL_H_car6), dim=1)
                DL_MU_H_label = torch.squeeze(DL_MU_H_label)
                H = DL_MU_H_label.cpu().numpy()

                SNR_ = 10**(SNR_dB / 10)
                noise_power = Pilot_power / SNR_  

                val_global_loss, global_loss_per_user, real_sumrate = loss_func(DL_MU_H_label, aggregated_precoder_output, noise_power, device)   
                val_global_loss_q, global_loss_per_user_q, real_sumrate_q = loss_func(DL_MU_H_label, aggregated_precoder_output_q, noise_power, device)  
                cnt_val += 1
                Val_loss += val_global_loss.item()
                Val_real_sumrate += real_sumrate.item()
                Val_loss_q += val_global_loss_q.item()
                Val_real_sumrate_q += real_sumrate_q.item()
            print(f'Val_loss (wo Quan): {Val_loss/cnt_val:.4f}')
            print(f'Sum rate (wo Quan): {Val_real_sumrate/cnt_val:.4f}')
            print('***************************')
            print(f'Val_loss (Quan): {Val_loss_q/cnt_val:.4f}')
            print(f'Sum rate (Quan): {Val_real_sumrate_q/cnt_val:.4f}')
            print('***************************')
            print(global_loss_per_user)
            

        Average.append(Val_real_sumrate/cnt_val)
        Average_q.append(Val_real_sumrate_q/cnt_val)
        print('----woQuant-----')
        print(np.mean(Average))
        print('----wQuant-----')
        print(np.mean(Average_q))

        for k in range(Num_user):
            output_description = vehicle_sensors_description(indicator[k])
            print(output_description)
        print('***********************************')
    print(np.mean(Average))
    print(np.mean(Average_q))
    print((Average_q))
    print('***************************')


if __name__ == "__main__":
    main()
       
