from src.utils.eval import *
from src.data import *
def get_data_triplet_file(data_path, output_path):
    dataset = TripletFaceDataset(data_path, 10, augment=False)
    with open(output_path, 'w') as f:
        for index in range(10000):
            a, p, n, a_label, p_label, n_label = dataset.get_anchor_positive_negative_paths(index)
            f.writelines(f'{a} {p} {n} {a_label} {p_label} {n_label}\n')



