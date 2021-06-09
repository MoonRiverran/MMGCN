import csv
import torch
import random

def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(md_data)

#若两个节点的值不等于0，则有一条边
def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


def data_pro(args):
    dataset = dict()
    dataset['md_p'] = read_csv(args.dataset_path + '/m_d.csv')
    dataset['md_true'] = read_csv(args.dataset_path + '/m_d.csv')

    zero_index = []
    one_index = []
    #将数据集的索引分成小于1和大于1的
    for i in range(dataset['md_p'].size(0)):
        for j in range(dataset['md_p'].size(1)):
            if dataset['md_p'][i][j] < 1:
                zero_index.append([i, j])
            if dataset['md_p'][i][j] >= 1:
                one_index.append([i, j])
    random.shuffle(one_index)  #将序列的所有元素随机排序
    random.shuffle(zero_index)
    zero_tensor = torch.LongTensor(zero_index)
    one_tensor = torch.LongTensor(one_index)
    dataset['md'] = dict()
    dataset['md']['train'] = [one_tensor, zero_tensor]


    "disease functional sim"
    dd_f_matrix = read_csv(args.dataset_path + '/d_d_f.csv')
    dd_f_edge_index = get_edge_index(dd_f_matrix)
    dataset['dd_f'] = {'data_matrix': dd_f_matrix, 'edges': dd_f_edge_index}

    "disease semantic sim"
    dd_s_matrix = read_csv(args.dataset_path + '/d_d_s.csv')
    dd_s_edge_index = get_edge_index(dd_s_matrix)
    dataset['dd_s'] = {'data_matrix': dd_s_matrix, 'edges': dd_s_edge_index}

    "miRNA functional sim"
    mm_f_matrix = read_csv(args.dataset_path + '/m_m_f.csv')
    mm_f_edge_index = get_edge_index(mm_f_matrix)
    dataset['mm_f'] = {'data_matrix': mm_f_matrix, 'edges': mm_f_edge_index}

    "miRNA sequence sim"
    mm_s_matrix = read_csv(args.dataset_path + '/m_m_s.csv')
    mm_s_edge_index = get_edge_index(mm_s_matrix)
    dataset['mm_s'] = {'data_matrix': mm_s_matrix, 'edges': mm_s_edge_index}

    return dataset

