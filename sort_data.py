from astropy.table import Table
import numpy as np
from tensorflow.keras.utils import to_categorical



# Sort through the data to only get 'real' data, no noise. For example the S/N must be larger than 3 
data = Table.read('data/data_matched_step2_newz_sm.csv',format='ascii.csv',header_start=0,data_start=1)

# ensure no NaN
valid_sigma_o3_err = (data['sigma_o3_err'] > 0) & np.isfinite(data['sigma_o3_err'])  # valid error values
valid_sigma_o3 = np.isfinite(data['sigma_o3'])  # valid sigma_o3 values
valid_data = valid_sigma_o3 & valid_sigma_o3_err  # only keep valid data points

ind1=np.where(np.array(data['o3']/data['o3_err']) >3)
ind2=np.where(np.array((data['o21']+data['o22'])/np.sqrt(data['o21_err']**2+data['o22_err']**2)) >3)
ind3=np.where(np.array(data['hb']/data['hb_err']) >3)
ind4=np.where(np.array(data['ha']/data['ha_err']) >3)
ind5=np.where(np.array(data['s21']/data['s21_err']) >3)
ind6 = np.where(valid_data & (data['sigma_o3'] / data['sigma_o3_err'] > 3))  # Only include valid data
ind7=np.where(np.array(data['sigma_o3']/data['sigma_o3_err']) >3)
ind8=np.where(np.array(data['VDISP'])>0.)
ind9=np.where(np.array(data['mag_u'])>10.)
ind10=np.where(np.array(data['flux_w1'])>0.)

ind_g=np.where(np.array(data['flux_g']) >0.)
ind_r=np.where(np.array(data['flux_r']) >0.)
ind_z=np.where(np.array(data['flux_z']) >0.)

# Calculate absolute magnitude (I think)
data['mag_g'][ind_g]=22.5-2.5*np.log10(data['flux_g'][ind_g])
data['mag_r'][ind_r]=22.5-2.5*np.log10(data['flux_r'][ind_r])
data['mag_z'][ind_z]=22.5-2.5*np.log10(data['flux_z'][ind_z])

# Sort through these indices
ind=np.array(list(set(ind1[0]) & set(ind1[0]) &set(ind2[0]) & set(ind3[0]) & set(ind4[0]) & set(ind5[0]) & set(ind6[0]) & set(ind7[0]) & set(ind8[0])))# & set(ind9[0])& set(ind10[0]) ))
n_source=len(ind)

# Split into training and test set
n_split=int(n_source*0.7)
ind_train=ind[0:n_split]
ind_test=ind[n_split:]
n_train=len(ind_train)
n_test=len(ind_test)
type_arr=np.zeros(len(ind))
type_arr=type_arr-999

# Get/define the 8 input features
z=np.array(data['z'][ind])
O2_index=np.log10((data['o21'][ind]+data['o22'][ind])/data['hb'][ind])
O3_index=np.log10(data['o3'][ind]/data['hb'][ind])
N2_index=np.log10(data['n2'][ind]/data['ha'][ind])
S2_index=np.log10((data['s21'][ind]+data['s22'][ind])/data['ha'][ind])
sigma_o3=np.log10(data['sigma_o3'][ind])
sigma_star=np.log10(data['VDISP'][ind])
u_g=data['mag_u'][ind]-data['mag_g'][ind]
g_r=data['mag_g'][ind]-data['mag_r'][ind]
r_i=data['mag_r'][ind]-data['mag_i'][ind]
i_z=data['mag_i'][ind]-data['mag_z'][ind]

# Correct the redshift
z_w1=data['mag_z'][ind]-(22.5-2.5*np.log10(data['flux_w1'][ind]))

# Classify the galaxies into one of the four classes: star-forming galaxies, composite galaxies, 
# active galactic nuclei (AGNs), or low-ionization nuclear emission regions (LINERs) 
ind_sf1=np.where(O3_index <= (0.61/(N2_index-0.05)+1.3))
ind_sf2=np.where( N2_index < 0.)
ind_sf=np.array(list(set(ind_sf1[0]) & set(ind_sf2[0])))

ind_AGN1=np.where(O3_index > (0.61/(N2_index-0.47)+1.19))
ind_AGN2=np.where(N2_index >= 0.)
ind_AGN3=np.where(O3_index > 1.89*S2_index+0.76)
ind_AGN=np.array(list((set(ind_AGN1[0]) | set(ind_AGN2[0])) & set(ind_AGN3[0])))


ind_liner1=np.where(O3_index > (0.61/(N2_index-0.47)+1.19))
ind_liner2=np.where(O3_index <= 1.89*S2_index+0.76)
ind_liner=np.array(list(set(ind_liner1[0]) & set(ind_liner2[0])))


ind_comp1=np.where(O3_index < (0.61/(N2_index-0.47)+1.19))
ind_comp2=np.where(O3_index > (0.61/(N2_index-0.05)+1.3))
ind_comp=np.array(list(set(ind_comp1[0]) & set(ind_comp2[0])))

type_arr[ind_sf]=1
type_arr[ind_comp]=2
type_arr[ind_AGN]=3
type_arr[ind_liner]=4

# Making the training classifications
type_arr_train=np.zeros(len(ind_train))
type_arr_train=type_arr_train-999

# Get/define the features for the training
O2_index_train=np.log10((data['o21'][ind_train]+data['o22'][ind_train])/data['hb'][ind_train])
O3_index_train=np.log10(data['o3'][ind_train]/data['hb'][ind_train])
N2_index_train=np.log10(data['n2'][ind_train]/data['ha'][ind_train])
S2_index_train=np.log10((data['s21'][ind_train]+data['s22'][ind_train])/data['ha'][ind_train])
sigma_o3_train=np.log10(data['sigma_o3'][ind_train])
sigma_star_train=np.log10(data['VDISP'][ind_train])
u_g_train=data['mag_u'][ind_train]-data['mag_g'][ind_train]
g_r_train=data['mag_g'][ind_train]-data['mag_r'][ind_train]
r_i_train=data['mag_r'][ind_train]-data['mag_i'][ind_train]
i_z_train=data['mag_i'][ind_train]-data['mag_z'][ind_train]
z_w1_train=data['mag_z'][ind_train]-(22.5-np.log10(data['flux_w1'][ind_train]))
sm_train=data['sm'][ind_train]

# Classification 
ind_sf1_train=np.where(O3_index_train <= (0.61/(N2_index_train-0.05)+1.3))
ind_sf2_train=np.where( N2_index_train < 0.)
ind_sf_train=np.array(list(set(ind_sf1_train[0]) & set(ind_sf2_train[0])))

ind_AGN1_train=np.where(O3_index_train > (0.61/(N2_index_train-0.47)+1.19))
ind_AGN2_train=np.where(N2_index_train >= 0.)
ind_AGN3_train=np.where(O3_index_train > 1.89*S2_index_train+0.76)
ind_AGN_train=np.array(list(set(ind_AGN1_train[0]) | set(ind_AGN2_train[0]) & set(ind_AGN3_train[0])))


ind_liner1_train=np.where(O3_index_train > (0.61/(N2_index_train-0.47)+1.19)) 
ind_liner2_train=np.where(O3_index_train <= 1.89*S2_index_train+0.76)
ind_liner_train=np.array(list(set(ind_liner1_train[0]) & set(ind_liner2_train[0])))


ind_comp1_train=np.where(O3_index_train < (0.61/(N2_index_train-0.47)+1.19))
ind_comp2_train=np.where(O3_index_train > (0.61/(N2_index_train-0.05)+1.3))
ind_comp_train=np.array(list(set(ind_comp1_train[0]) & set(ind_comp2_train[0])))


type_arr_train[ind_sf_train]=1
type_arr_train[ind_comp_train]=2
type_arr_train[ind_AGN_train]=3
type_arr_train[ind_liner_train]=4

# Testing features + classifications
type_arr_test=np.zeros(len(ind_test))
type_arr_test=type_arr_test-999

O2_index_test=np.log10((data['o21'][ind_test]+data['o22'][ind_test])/data['hb'][ind_test])
O3_index_test=np.log10(data['o3'][ind_test]/data['hb'][ind_test])
N2_index_test=np.log10(data['n2'][ind_test]/data['ha'][ind_test])
S2_index_test=np.log10((data['s21'][ind_test]+data['s22'][ind_test])/data['ha'][ind_test])
sigma_o3_test=np.log10(data['sigma_o3'][ind_test])
sigma_star_test=np.log10(data['VDISP'][ind_test])
u_g_test=data['mag_u'][ind_test]-data['mag_g'][ind_test]
g_r_test=data['mag_g'][ind_test]-data['mag_r'][ind_test]
r_i_test=data['mag_r'][ind_test]-data['mag_i'][ind_test]
i_z_test=data['mag_i'][ind_test]-data['mag_z'][ind_test]
z_w1_test=data['mag_z'][ind_test]-(22.5-np.log10(data['flux_w1'][ind_test]))
sm_test=data['sm'][ind_test]

ind_sf1_test=np.where(O3_index_test <= (0.61/(N2_index_test-0.05)+1.3))
ind_sf2_test=np.where( N2_index_test < 0.)
ind_sf_test=np.array(list(set(ind_sf1_test[0]) & set(ind_sf2_test[0])))

ind_AGN1_test=np.where(O3_index_test > (0.61/(N2_index_test-0.47)+1.19))
ind_AGN2_test=np.where(N2_index_test >= 0.)
ind_AGN3_test=np.where(O3_index_test > 1.89*S2_index_test+0.76)
ind_AGN_test=np.array(list(set(ind_AGN1_test[0]) | set(ind_AGN2_test[0]) & set(ind_AGN3_test[0])))


ind_liner1_test=np.where(O3_index_test > (0.61/(N2_index_test-0.47)+1.19))
ind_liner2_test=np.where(O3_index_test <= 1.89*S2_index_test+0.76)
ind_liner_test=np.array(list(set(ind_liner1_test[0]) & set(ind_liner2_test[0])))


ind_comp1_test=np.where(O3_index_test < (0.61/(N2_index_test-0.47)+1.19))
ind_comp2_test=np.where(O3_index_test > (0.61/(N2_index_test-0.05)+1.3))
ind_comp_test=np.array(list(set(ind_comp1_test[0]) & set(ind_comp2_test[0])))

type_arr_test[ind_sf_test]=1
type_arr_test[ind_comp_test]=2
type_arr_test[ind_AGN_test]=3
type_arr_test[ind_liner_test]=4

ind_this=ind_sf

# Creating a dictionary with all X values
features_train = {
        'O2_index': np.log10((data['o21'][ind_train] + data['o22'][ind_train]) / data['hb'][ind_train]),
        'O3_index': np.log10(data['o3'][ind_train] / data['hb'][ind_train]),
        'sigma_o3': np.log10(data['sigma_o3'][ind_train]),
        'sigma_star': np.log10(data['VDISP'][ind_train]),
        'u_g': data['mag_u'][ind_train] - data['mag_g'][ind_train],
        'g_r': data['mag_g'][ind_train] - data['mag_r'][ind_train],
        'r_i': data['mag_r'][ind_train] - data['mag_i'][ind_train],
        'i_z': data['mag_i'][ind_train] - data['mag_z'][ind_train],
    }

features_test = {
        'O2_index': np.log10((data['o21'][ind_test] + data['o22'][ind_test]) / data['hb'][ind_test]),
        'O3_index': np.log10(data['o3'][ind_test] / data['hb'][ind_test]),
        'sigma_o3': np.log10(data['sigma_o3'][ind_test]),
        'sigma_star': np.log10(data['VDISP'][ind_test]),
        'u_g': data['mag_u'][ind_test] - data['mag_g'][ind_test],
        'g_r': data['mag_g'][ind_test] - data['mag_r'][ind_test],
        'r_i': data['mag_r'][ind_test] - data['mag_i'][ind_test],
        'i_z': data['mag_i'][ind_test] - data['mag_z'][ind_test],
    }

# Create the final arrays from the dictionaries
X_train = np.column_stack([features_train['O2_index'], features_train['O3_index'], features_train['sigma_o3'],
                           features_train['sigma_star'], features_train['u_g'], features_train['g_r'], 
                           features_train['r_i'], features_train['i_z']])

X_test = np.column_stack([features_test['O2_index'], features_test['O3_index'], features_test['sigma_o3'],
                          features_test['sigma_star'], features_test['u_g'], features_test['g_r'],
                          features_test['r_i'], features_test['i_z']])

valid_train_indices = type_arr_train != -999
X_train = X_train[valid_train_indices]
type_arr_train = type_arr_train[valid_train_indices] 

valid_test_indices = type_arr_test != -999
X_test = X_test[valid_test_indices]
type_arr_test = type_arr_test[valid_test_indices] 

# One hot encoding of the labels
y_train = to_categorical(type_arr_train - 1, num_classes=4)
y_test = to_categorical(type_arr_test - 1, num_classes=4)

np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy', X_test)
np.save('data/y_train.npy',y_train)
np.save('data/y_test.npy', y_test)

sizes = [len(ind_sf_train)+len(ind_sf_test), len(ind_comp_train) + len(ind_comp_test), len(ind_AGN_train) + len(ind_AGN_test), len(ind_liner_train)+ len(ind_liner_test)]
np.save('data/sizes.npy', sizes)