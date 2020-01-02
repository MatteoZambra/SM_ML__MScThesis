
"""
Taken from https://stackoverflow.com/a/33720100/9136498
slight modification done

Minimal documentation, I only used this code _as it is_, for
the purpose of plotting the network with connection strengths 
made visually clear.
"""

from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np

pyplot.style.use('default')

vertical_distance_between_layers = 30
horizontal_distance_between_neurons = 2
neuron_radius = 0.5
number_of_neurons_in_widest_layer = 31


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), 
                               radius=neuron_radius, 
                               color = 'k',
                               fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, weights):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, _linewidth, asGraph):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        if (asGraph == False):
            if (_linewidth <= 0):
                line = pyplot.Line2D(line_x_data, 
                                     line_y_data, 
                                     color='r',
                                     linewidth=_linewidth*1.5)
                                     #linewidth=0.3)
            else:
                line = pyplot.Line2D(line_x_data, 
                                     line_y_data, 
                                     color='g',
                                     linewidth=_linewidth*1.5)
                                     #linewidth=0.3)
            #end
        else:
            line = pyplot.Line2D(line_x_data,
                                 line_y_data,
                                 color = 'b',
                                 linewidth = 0.3*_linewidth)
        #end            
        pyplot.gca().add_line(line)
    def draw(self, asGraph):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight, asGraph)


class NeuralNetwork():
    def __init__(self, asGraph = False):
        self.layers = []
        self.asGraph = asGraph

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self, path):
        for layer in self.layers:
            layer.draw(self.asGraph)
        pyplot.axis('scaled')
        pyplot.savefig(path, dpi=300, bbox_inches = "tight")
        pyplot.show()



class plotNet():
    def __init__(self, weights, path, trained, asGraph):
        """
        maybe it is worthless to fill the class with these
        things, would it be better to pass some of the arguments
        to the plotNetFunction directly?
        """
        
        _weights = np.asarray(weights)

        numLayers = int(_weights.shape[0]/2)
        wghts = []
        biases = []

        for i in range(numLayers):
            j = 2*i
#            print(j,(_weights[j].T).shape)
            wghts.append(_weights[j])
            j = 2*i + 1
#            print(j,(_weights[j].T).shape)
            biases.append(_weights[j])
        #enddo

        self.numLayers = numLayers
        self.wghts = np.asarray(wghts)
        self.asGraph = asGraph
        self.wghts = wghts
        self.path = path
        self.trained = trained
    #end

    def plotNetFunction(self):
        
        print("NetPlot function")
        network = NeuralNetwork(asGraph = self.asGraph)
        # weights to convert from 10 outputs to 4 (decimal digits to their binary representation)

        pyplot.figure(figsize=(8,30))
        fig = pyplot.gca()
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)


        for i in range(self.numLayers-1):
            wghs = self.wghts[i]
            network.add_layer((wghs.T).shape[1], wghs.T)
        #enddo
        network.add_layer((self.wghts[-1].T).shape[1], self.wghts[-1].T)
        network.add_layer((self.wghts[-1].T).shape[0])
        
        if (self.trained):
            self.path += r'net_trained.png'
        else:
            self.path += r'net_untrained.png'

        network.draw(self.path)
#        pyplot.savefig(self.path, dpi=300, bbox_inches = "tight")
#        pyplot.show()