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
    input_nodes = int(sys.argv[4])
    layer_number = int(sys.argv[5])
    layer_number_adjust = int(sys.argv[6])
    layer_placehold = int(sys.argv[7])
    layers_loss = float(sys.argv[8])
    lr = training_new.lr_scheduler()
    nodes = int(sys.argv[10])
    layer_placehold, layer_loss = training_new.layer_hyp(cells, datas, epochs, input_nodes, layer_number, layer_number_adjust, layer_placehold, layers_loss, lr, nodes)
    print(layer_placehold)
    print(layer_loss)
    

if __name__ == "__main__":
    main()
