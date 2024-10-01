import matplotlib.pyplot as plt
import numpy as np
import json
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def save_np_json(np_data, filename):
    with open(filename, "w") as j_file:
        json.dump(np_data, j_file, cls=NumpyArrayEncoder, indent=2)
        print('----->>> json file save!! \'{}\''.format(filename))


def load_np_json(filename, keyword):
    with open(filename, "r") as j_file:
        load_file = json.load(j_file)
        np_file = np.asarray(load_file[keyword])
        return np_file
        


# x = np.array([[458, 86], [451, 164], [287, 181],
#         [196, 383], [297, 444], [564, 194],
#         [562, 375], [596, 520], [329, 620],
#         [488, 622], [432, 52], [489, 56]])

# y = np.array([[540, 311], [603, 359], [542, 378],
#         [525, 507], [485, 542], [691, 352],
#         [752, 488], [711, 605], [549, 651],
#         [651, 663], [526, 293], [542, 290]])
# # plt.scatter(x[:, 0], x[:, 1])
# plt.scatter(y[:, 0], y[:, 1])
# plt.show()

# # 创建两个矩阵
# mtx1 = np.array([[1, 2], [3, 4]])
# RT1 = np.array([[0.5, 0], [0, 2]])

# # 执行矩阵乘法
# P1 = mtx1 @ RT1
# print(mtx1)
# print(RT1)
# print(P1)

if __name__ == '__main__':
    # label = np.array([
    #     [345, 27],
    #     [327, 71],
    #     [356, 71],
    #     [387, 126],
    #     [451, 149],
    #     [285, 97],
    #     [249, 186],
    #     [268, 254],
    #     [346, 211],
    #     [368, 317],
    #     [355, 371],
    #     [325, 345],
    #     [330, 408]
    #   ])
    # data = {'joints': label}
    # save_np_json(data, 'cam3_000015.json')
    
    save_path = './videos_pair/{}.json'.format("video_1692338495")

    keypoints = load_np_json(save_path, keyword='keypoints')
    print(keypoints)