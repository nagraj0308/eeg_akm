from common.tools import create_matrix_file, show_results
from config import CHANNEL
from experments.single_channel import single_channel

create_matrix_file(CHANNEL)
single_channel(CHANNEL)
show_results(CHANNEL)
