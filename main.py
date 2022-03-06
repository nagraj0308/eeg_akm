from common.tools import create_matrix_file
from experments.single_channel import single_channel
import numpy as np
import mne


create_matrix_file()
single_channel()



# edf = mne.io.read_raw_edf('C:/Users/Admin/Downloads/chb01_01.edf')
# header = ','.join(edf.ch_names)
# np.savetxt('C:/Users/Admin/Downloads/chb01_01.csv', edf.get_data().T, delimiter=',', header=header)
