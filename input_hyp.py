import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import training_new


def main():
    cells = int(sys.argv[1])
    strOutputFile = sys.argv[2]
    datas = pickle.load(open(strOutputFile, 'rb'))
    epochs = int(sys.argv[3])
    input_loss = float(sys.argv[4])
    input_nodes = int(sys.argv[5])
    input_nodes_adjust = int(sys.argv[6])
    input_placehold = int(sys.argv[7])
    layer_number = int(sys.argv[8])
    lr = training_new.lr_scheduler()
    nodes = int(sys.argv[10])
    normal_file = sys.argv[11]
    normalisation_layer = pickle.load(open(normal_file))
    input_loss, input_placehold = training_new.input_hyp(normalisation_layer, cells, datas, epochs, input_loss, input_nodes, input_nodes_adjust, input_placehold, layer_number, lr, nodes)
    print(input_placehold)
    print(input_loss)
    

if __name__ == "__main__":
    main()
