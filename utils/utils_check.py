# Author   : Liu YuHang
# Date     : 2022/10/30
# Time     : 10:40
# Function :

import torch
import numpy as np



if __name__ == '__main__':
    input = [-4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4]
    input = np.array(input)
    input = torch.from_numpy(input)

    print(torch.sigmoid(input))
    # [0.0180, 0.0474, 0.1192, 0.2689, 0.3775, 0.5000, 0.6225, 0.7311, 0.8808, 0.9526, 0.9820]

    # 加法调整系数的话，较为合理的取值可能为【-1,1】，调整影响【0.27,0.73】间概率


    # yolo = YOLO()
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # input = torch.rand([7, 3, 416, 416]).to(device)
    #
    # # 邻接矩阵，转换为tensor后需指定为torch.float32
    # adj = [
    #     [0, 1, 1],
    #     [1, 0, 1],
    #     [1, 1, 0],
    # ]
    # # torch.float32
    # adj = torch.tensor(adj, dtype=torch.float).to(device)
    #
    # anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # num_classes = 3
    # phi = 's'
    # model = YoloBody(anchors_mask, num_classes, phi, adj, input_shape=[416, 416])
    # model.to(device)
    # #out0, out1, out2, out0_fix, out1_fix, out2_fix, meta = model(input)
    #
    # outputs = model(input)[:3]
    # outputs = yolo.bbox_util.decode_box(outputs)
    # print(outputs.shape)

    # check(out0, out1, out2)

    # print(out0.shape)
    # print(out1.shape)
    # print(out2.shape)
    # print(out0_fix.shape)
    # print(out1_fix.shape)
    # print(out2_fix.shape)
    #
    # print('meta:', meta)