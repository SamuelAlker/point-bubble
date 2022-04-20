import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import training_new


def main():
    cell_placehold = int(sys.argv[1])
    cells = int(sys.argv[2])
    cells_adjust = int(sys.argv[3])
    cells_loss = float(sys.argv[4])
    strOutputFile = sys.argv[5]
    datas = pickle.load(open(strOutputFile, 'rb'))
    epochs = int(sys.argv[6])
    input_nodes = int(sys.argv[7])
    layer_number = int(sys.argv[8])
    lr = training_new.lr_scheduler()
    nodes = int(sys.argv[10])
    normal_file = sys.argv[11]
    normalisation_layer = pickle.load(open(normal_file))
    cell_placehold, cells_loss = training_new.cells_hyp(normalisation_layer, cell_placehold, cells, cells_adjust, cells_loss, datas, epochs, input_nodes, layer_number, lr, nodes)
    print(cell_placehold)
    print(cells_loss)
    

if __name__ == "__main__":
    main()
