import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import training_new


def main():
    input_nodes = int(sys.argv[1])
    nodes = int(sys.argv[2])
    layer_number = int(sys.argv[3])
    cells = int(sys.argv[4])
    strOutputFile = sys.argv[5]
    datas = pickle.load(open(strOutputFile, 'rb'))
    test_epochs = int(sys.argv[6])
    lr = training_new.lr_scheduler()
    previous_loss = float(sys.argv[8])
    directory = str(sys.argv[9])
    normal_file = sys.argv[10]
    normalisation_layer = pickle.load(open(normal_file))
    break_point = training_new.test_model(normalisation_layer, input_nodes, nodes, layer_number, cells, datas, test_epochs, lr, previous_loss, directory)
    print(break_point)


if __name__ == "__main__":
    main()
