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
    lr = training_new.lr_scheduler()
    node_placehold = int(sys.argv[7])
    nodes = int(sys.argv[8])
    nodes_adjust = int(sys.argv[9])
    nodes_loss = float(sys.argv[10])
    node_placehold, node_loss = training_new.node_hyp(cells, datas, epochs, input_nodes, layer_number, lr, node_placehold, nodes, nodes_adjust, nodes_loss)
    print(node_placehold)
    print(node_loss)
    

if __name__ == "__main__":
    main()
