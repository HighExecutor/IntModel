import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def m2r(x, y, x_size):
    return x * x_size + y


if __name__ == '__main__':
    path = "C:\\wspace\\projects\\IntModel\\resources\\marsh.csv"
    data = pd.read_csv(path, delimiter=';')
    ids = data.id_card.unique()

    x_max = data['lon'].max()
    x_min = data['lon'].min()
    y_max = data['lat'].max()
    y_min = data['lat'].min()
    x_b_min = 3300000
    x_b_max = 3420000
    x_b_diff = x_b_max - x_b_min
    y_b_min = 8320000
    y_b_max = 8440000
    y_b_diff = y_b_max - y_b_min
    x_step = 4000
    y_step = 4000
    x_size = int(x_b_diff / x_step)
    y_size = int(y_b_diff / y_step)
    cells = x_size * y_size
    print("x_size = {}, y_size = {}, cells = {}".format(x_size, y_size, cells))
    routes = list()
    i = 0
    for id in ids:
        # if i == 50000:
        #     break
        print(i)
        i += 1
        id_data = data[data['id_card'] == id]
        id_x = list()
        id_y = list()
        # plt.figure()
        route = list()
        for _, trans in id_data.iterrows():
            t_time_str = trans['time']
            t_time = time.strptime(t_time_str, '%H:%M')
            t_abs = t_time.tm_hour * 60 + t_time.tm_min
            x_pos = trans['lon']
            y_pos = trans['lat']
            if x_pos > x_b_min and x_pos < x_b_max and y_pos > y_b_min and y_pos < y_b_max:
                # plt.plot(x_pos, y_pos, marker="o")
                # plt.text(x_pos, y_pos, t_time)
                route.append((x_pos, y_pos, t_abs))
                id_x.append(x_pos)
                id_y.append(y_pos)
        if len(id_x) > 1:
            routes.append(route)
            # plt.plot(id_x, id_y)
        # plt.xlim(x_b_min, x_b_max)
        # plt.ylim(y_b_min, y_b_max)
        # plt.title(str(id))
        # plt.show()
        # plt.close()

    spb_transporations = np.zeros((1440, cells, cells), dtype=np.int32)

    for route in routes:
        r_len = len(route)
        cur_x = -1
        cur_y = -1
        cur_x_c = -1
        cur_y_c = -1
        cur_c = -1
        for i in range(r_len - 1):
            p_from = route[i]
            p_to = route[i + 1]
            x_diff = p_to[0] - p_from[0]
            y_diff = p_to[1] - p_from[1]
            t_diff = p_to[2] - p_from[2]
            x_diff_step = np.floor(x_diff / t_diff)
            y_diff_step = np.floor(y_diff / t_diff)
            for i_t in range(p_from[2], p_to[2]):
                if i == 0:
                    cur_x = p_from[0]
                    cur_y = p_from[1]
                    cur_x_c = int(np.floor((cur_x - x_b_min) / x_step))
                    cur_y_c = int(np.floor((cur_y - y_b_min) / y_step))
                    cur_c = m2r(cur_x_c, cur_y_c, x_size)
                    if cur_c > 900:
                        raise Exception
                    spb_transporations[i_t][cur_c][cur_c] += 1
                else:
                    cur_x += x_diff_step
                    cur_y += y_diff_step
                    cur_x_c = int(np.floor((cur_x - x_b_min) / x_step))
                    cur_y_c = int(np.floor((cur_y - y_b_min) / y_step))
                    new_c = m2r(cur_x_c, cur_y_c, x_size)
                    if new_c > 900:
                        raise Exception
                    if cur_c != new_c:
                        spb_transporations[i_t][cur_c][new_c] += 1
                    cur_c = new_c
            cur_x = p_to[0]
            cur_y = p_to[1]
            pass
    #
    # trans_file = open("C:\\wspace\\projects\\IntModel\\resources\\spb_passengers", 'w')
    # trans_file.write("1440\n")
    # for i in range(1440):
    #     print("write {}".format(i))
    #     not_zero_cells = np.count_nonzero(spb_transporations[i])
    #     trans_file.write(str(not_zero_cells) + "\n")
    #     for j in range(cells):
    #         for k in range(cells):
    #             if spb_transporations[i][j][k] != 0:
    #                 print("{}\t{}\t{}".format(j, k, spb_transporations[i][j][k]))
    #                 trans_file.write("{}\t{}\t{}\n".format(j, k, spb_transporations[i][j][k]))
    # trans_file.close()
    print("Max value = {}".format(spb_transporations.max()))
    # print("x_size = {}, y_size = {}, cells = {}".format(x_size, y_size, cells))
