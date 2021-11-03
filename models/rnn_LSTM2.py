import torch
import torch.nn as nn
"""TODO
Try with a different dataset of category -> line, for example:
    Fictional series -> Character name
    Part of speech -> Word
    Country -> City
Use a “start of sentence” token so that sampling can be done without choosing a start letter
Get better results with a bigger and/or better shaped network
    Try the nn.LSTM and nn.GRU layers
    Combine multiple of these RNNs as a higher level network
"""

class RNN(nn.Module):
    """
    RNN with an extra argument for the category tensor, which is concatenated along with the others. 
    The category tensor is a one-hot vector just like the letter input.
    
    Interpretation: 
        We will interpret the output as the probability of the next letter. 
    When sampling, the most likely output letter is used as the next input letter.
    
    Layers:
    * A second linear layer o2o (after combining hidden and output) to give it more muscle to work with. 
    * There’s also a dropout layer, which randomly zeros parts of its input with a given probability (here 0.1)
      It is usually used to fuzz inputs to prevent overfitting. 
      Here we purposely add some chaos and increase sampling variety.
    """
    def __init__(self, n_categories, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.i2h = nn.LSTM(n_categories + input_size + hidden_size, hidden_size, batch_first=True)
        self.i2o = nn.LSTM(n_categories + input_size + hidden_size, output_size, batch_first=True)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.hhn = torch.zeros(1, 1, hidden_size)
        self.hcn = torch.zeros(1, 1, hidden_size)
        self.ohn = torch.zeros(1, 1, output_size)
        self.ocn = torch.zeros(1, 1, output_size)
        

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        input_combined = input_combined.unsqueeze(0)
        hidden, (hhn, hcn) = self.i2h(input_combined, (self.hhn, self.hcn))
        output, (ohn, ocn) = self.i2o(input_combined, (self.ohn, self.ocn))
        hidden, output = hidden[0], output[0]
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

   
