# DTFL
This repository contains the code for the paper "Speed Up Federated Learning in Heterogeneous Environment: A Dynamic Tiering Approach"

DTFL is a federated learning algorithm designed to speed up training in heterogeneous environments. In heterogeneous environments, the devices participating in federated learning may have different computing resources and network connectivity. DTFL addresses this by dynamically assigning devices to different tiers based on their capabilities.

To do this, this Python program simulates the CPU profile for each client and calculates the intermediate data size and delay. It then uses this information to assign clients to tiers to minimize the overall training time.

DTFL has been shown to achieve significant speedups over traditional federated learning algorithms in heterogeneous environments.

The dataset can be downloaded by running the following command:

"
sh download_dataset.sh
"

## Usage
To train the model, run the following command:

'''
python3 main.py
'''
