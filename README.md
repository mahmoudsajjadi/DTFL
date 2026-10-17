# DTFL
This repository contains the code for the paper "Speed Up Federated Learning in Heterogeneous Environments: A Dynamic Tiering Approach"

DTFL is a federated learning algorithm designed to speed up training in heterogeneous environments. In heterogeneous environments, the devices participating in federated learning may have different computing resources and network connectivity. DTFL addresses this by dynamically assigning devices to different tiers based on their capabilities.

To do this, this Python program simulates the CPU profile for each client and calculates the intermediate data size and delay. It then uses this information to assign clients to tiers to minimize the overall training time. Detailed information about training time and each client's performance is logged on Weights & Biases ([WandB](https://wandb.ai/)).

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

---

## 📄 Related Research & Citations

If you find this repository useful, please also check out our recent peer-reviewed publications and benchmarks in distributed learning, multi-agent systems, and smart energy grids:

- **[Sustainable Smart Grid Resilience and Renewable Integration Using Multi-Agent Reinforcement Learning](https://www.mdpi.com/2071-1050/18/20/10574)** — *Seyed Mahmoud Sajjadi Mohammadabadi, Sayed Mohammad Sajjadi*, *MDPI Sustainability*, Vol. 18, Issue 20, 10574 (2026). [[MDPI Open Access](https://www.mdpi.com/2071-1050/18/20/10574)] • [[DOI: 10.3390/su182010574](https://doi.org/10.3390/su182010574)] • [[Code & Benchmarks](https://github.com/mahmoudsajjadi/Smart-Grid-Resilience-MARL)] • [[Interactive Page & BibTeX](https://mahmoudsajjadi.github.io/papers/sustainable-smart-grid-resilience-marl.html)]
- **[Speed Up Federated Learning in Heterogeneous Environments: A Dynamic Tiering Approach](https://doi.org/10.1109/JIOT.2024.3468511)** — *Seyed Mahmoud Sajjadi Mohammadabadi, Syed Zawad, Feng Yan, Lei Yang*, *IEEE Internet of Things Journal*, Vol. 12, No. 5, pp. 5026–5035 (2025). [[DOI](https://doi.org/10.1109/JIOT.2024.3468511)] • [[Code](https://github.com/mahmoudsajjadi/DTFL)]
- **[Communication-Efficient Training Workload Balancing for Decentralized Multi-Agent Learning](https://doi.org/10.1109/ICDCS60910.2024.00062)** — *Seyed Mahmoud Sajjadi Mohammadabadi, Lei Yang, Feng Yan, Junshan Zhang*, *IEEE ICDCS* (2024). [[DOI](https://doi.org/10.1109/ICDCS60910.2024.00062)] • [[Code](https://github.com/mahmoudsajjadi/ComDML)]
- **[Towards Distributed Learning of PMU Data: A Federated Learning Based Event Classification Approach](https://doi.org/10.1109/PESGM52003.2023.10252874)** — *Seyed Mahmoud Sajjadi Mohammadabadi, Yunchuan Liu, Abraham Canafe, Lei Yang*, *IEEE PESGM* (2023). [[DOI](https://doi.org/10.1109/PESGM52003.2023.10252874)]

```bibtex
@article{sajjadi2026sustainable,
  title     = {Sustainable Smart Grid Resilience and Renewable Integration Using Multi-Agent Reinforcement Learning},
  author    = {Sajjadi Mohammadabadi, Seyed Mahmoud and Sajjadi, Sayed Mohammad},
  journal   = {Sustainability},
  volume    = {18},
  number    = {20},
  pages     = {10574},
  year      = {2026},
  publisher = {MDPI},
  doi       = {10.3390/su182010574},
  url       = {https://www.mdpi.com/2071-1050/18/20/10574}
}
```

> 🏛️ Explore all 21 publications and 1-click BibTeX citations at the [**Academic Research Hub**](https://mahmoudsajjadi.github.io/research-hub.html) and [**Google Scholar**](https://scholar.google.com/citations?user=tkhCR5IAAAAJ&hl=en).