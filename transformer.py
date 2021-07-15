from batch_iterator import BatchIterator
from early_stopping import EarlyStopping
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import device
import math
import tqdm
import numpy as np
# import CuPy as cp
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter

CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if CUDA else 'cpu')

torch.cuda.device(device)

class MultiHeadAttention(nn.Module):
    """Implementation of the Multi-Head-Attention.

    Parameters
    ----------
    dmodel: int
        Dimensionality of the input embedding vector.
    heads: int
        Number of the self-attention operations to conduct in parallel.
    """

    def __init__(self, dmodel, heads):
        super(MultiHeadAttention, self).__init__()
        self.to(device)

        assert dmodel % heads == 0, 'Embedding dimension is not divisible by number of heads'

        self.dmodel = dmodel
        self.heads = heads
        self.key_dim = dmodel // heads

        self.linear = nn.ModuleList([
            nn.Linear(self.dmodel, self.dmodel, bias=False),
            nn.Linear(self.dmodel, self.dmodel, bias=False),
            nn.Linear(self.dmodel, self.dmodel, bias=False)])

        self.concat = nn.Linear(self.dmodel, self.dmodel, bias=False)

    def forward(self, inputs):
        """ Perform Multi-Head-Attention.

        Parameters
        ----------
        inputs: torch.Tensor
            Batch of inputs - position encoded word embeddings ((batch_size, seq_length, embedding_dim)

        Returns
        -------
        torch.Tensor
            Multi-Head-Attention output of a shape (batch_size, seq_len, dmodel)
        """

        self.batch_size = inputs.size(0)

        assert inputs.size(2) == self.dmodel, 'Input sizes mismatch, dmodel={}, while embedd={}'.format(self.dmodel,
                                                                                                        inputs.size(2))
        query, key, value = [linear(x).view(self.batch_size, -1, self.heads, self.key_dim).transpose(1, 2) for linear, x
                             in zip(self.linear, (inputs, inputs, inputs))]

        score = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.key_dim)

        soft_score = F.softmax(score, dim=-1)
        out = torch.matmul(soft_score, value).transpose(1, 2).contiguous().view(self.batch_size, -1,
                                                                                self.heads * self.key_dim)
        
        maximum_value = torch.max(out)#torch.FloatTensor([math.sqrt(torch.max(value))]).to(args.device)
        if maximum_value > 1.0:
            maximum_value = torch.FloatTensor([math.sqrt(torch.max(out))]).to(device)
            out.divide_(maximum_value)
            
        out = self.concat(out)
        return out

class PositionalEncoding(nn.Module):
    """Implementation of the positional encoding.

    Parameters
    ----------
    max_len: int
        The maximum expected sequence length.
    dmodel: int
        Dimensionality of the input embedding vector.
    dropout: float
        Probability of an element of the tensor to be zeroed.
    padding_idx: int
        Index of the padding token in the vocabulary and word embedding.

    """

    def __init__(self, max_len, dmodel, dropout, padding_idx):
        super(PositionalEncoding, self).__init__()
        self.to(device)

        self.dropout = nn.Dropout(dropout)

        self.pos_encoding = torch.zeros(max_len, dmodel).to(device)
        positions = torch.repeat_interleave(torch.arange(float(max_len)).unsqueeze(1), dmodel, dim=1)
        dimensions = torch.arange(float(dmodel)).repeat(max_len, 1)

        trig_fn_arg = positions / (torch.pow(10000, 2 * dimensions / dmodel))

        self.pos_encoding[:, 0::2] = torch.sin(trig_fn_arg[:, 0::2])
        self.pos_encoding[:, 1::2] = torch.cos(trig_fn_arg[:, 1::2])

        if padding_idx:
            self.pos_encoding[padding_idx] = 0.0

        self.pos_encoding = self.pos_encoding.unsqueeze(0)

    def forward(self, embedd):
        """Apply positional encoding.

        Parameters
        ----------
        embedd: torch.Tensor
            Batch of word embeddings ((batch_size, seq_length, dmodel = embedding_dim))

        Returns
        -------
        torch.Tensor
            Sum of word embeddings and positional embeddings (batch_size, seq_length, dmodel)
        """
        embedd = embedd + self.pos_encoding[:, :embedd.size(1), :]
        embedd = self.dropout(embedd)

        return embedd

class LabelSmoothingLoss(nn.Module):
    """Implementation of label smoothing with the Kullback-Leibler divergence Loss.

    Example:
    label_smoothing/(output_size-1) = 0.1
    confidence = 1 - 0.1 = 0.9

    True labels      Smoothed one-hot labels
        |0|              [0.9000, 0.1000]
        |0|              [0.9000, 0.1000]
        |1|              [0.1000, 0.9000]
        |1|    label     [0.1000, 0.9000]
        |0|  smoothing   [0.9000, 0.1000]
        |1|    ---->     [0.1000, 0.9000]
        |0|              [0.9000, 0.1000]
        |0|              [0.9000, 0.1000]
        |0|              [0.9000, 0.1000]
        |1|              [0.1000, 0.9000]

    Parameters
    ----------
    output_size: int
         The number of classes.
    label_smoothing: float, optional (default=0)
        The smoothing parameter. Takes the value in range [0,1].

    """

    def __init__(self, output_size, label_smoothing=0):
        super(LabelSmoothingLoss, self).__init__()
        self.to(device)

        self.output_size = output_size
        self.label_smoothing = label_smoothing
        self.confidence = 1 - self.label_smoothing

        assert label_smoothing >= 0.0 and label_smoothing <= 1.0, 'Label smoothing parameter takes values in the range [0, 1]'

        self.criterion = nn.KLDivLoss()

    def forward(self, pred, target):
        """Smooth the target labels and calculate the Kullback-Leibler divergence loss.

        Parameters
        ----------
        pred: torch.Tensor
            Batch of log-probabilities (batch_size, output_size)
        target: torch.Tensor
            Batch of target labels (batch_size, seq_length)

        Returns
        -------
        torch.Tensor
            The Kullback-Leibler divergence Loss.

        """
        one_hot_probs = torch.full(size=pred.size(), fill_value=self.label_smoothing / (self.output_size - 1)).to(device)
        one_hot_probs.scatter_(1, target.unsqueeze(1), self.confidence)

        return self.criterion(pred, one_hot_probs)

class TransformerBlock(nn.Module):
    """Implementation of single Transformer block.

    Transformer block structure:
    x --> Multi-Head --> Layer normalization --> Pos-Wise FFNN --> Layer normalization --> y
      |   Attention   |                       |                 |
      |_______________|                       |_________________|
     residual connection                      residual connection

    Parameters
    ----------
    dmodel: int
        Dimensionality of the input embedding vector.
    ffnn_hidden_size: int
        Position-Wise-Feed-Forward Neural Network hidden size.
    heads: int
        Number of the self-attention operations to conduct in parallel.
    dropout: float
        Probability of an element of the tensor to be zeroed.
    """

    def __init__(self, dmodel, ffnn_hidden_size, heads, dropout):
        super(TransformerBlock, self).__init__()
        self.to(device)

        self.attention = MultiHeadAttention(dmodel, heads).to(device)
        self.layer_norm1 = nn.LayerNorm(dmodel)
        self.layer_norm2 = nn.LayerNorm(dmodel)

        self.ffnn = nn.Sequential(
            nn.Linear(dmodel, ffnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffnn_hidden_size, dmodel))

    def forward(self, inputs):
        """Forward propagate through the Transformer block.

        Parameters
        ----------
        inputs: torch.Tensor
            Batch of embeddings.

        Returns
        -------
        torch.Tensor
            Output of the Transformer block (batch_size, seq_length, dmodel)
        """
        output = inputs + self.attention(inputs)
        output = self.layer_norm1(output)
        #output = output + self.ffnn(output)
        ffn_output = self.ffnn(output)
        
        maximum_value = torch.max(ffn_output)#torch.FloatTensor([math.sqrt(torch.max(value))]).to(args.device)
        if maximum_value > 1.0:
            maximum_value = torch.FloatTensor([math.sqrt(torch.max(ffn_output))]).to(device)
            ffn_output.divide_(maximum_value)
        
        output = output + ffn_output
        output = self.layer_norm2(output)

        return output


class Transformer(nn.Module):
    """Implementation of the Transformer model for classification.

    Parameters
    ----------
    vocab_size: int
        The size of the vocabulary.
    dmodel: int
        Dimensionality of the embedding vector.
    max_len: int
        The maximum expected sequence length.
    padding_idx: int, optional (default=0)
        Index of the padding token in the vocabulary and word embedding.
    n_layers: int, optional (default=4)
        Number of the stacked Transformer blocks.
    ffnn_hidden_size: int, optonal (default=dmodel * 4)
        Position-Wise-Feed-Forward Neural Network hidden size.
    heads: int, optional (default=8)
        Number of the self-attention operations to conduct in parallel.
    pooling: str, optional (default='max')
        Specify the type of pooling to use. Available options: 'max' or 'avg'.
    dropout: float, optional (default=0.2)
        Probability of an element of the tensor to be zeroed.
    """

    def __init__(self):
        super(Transformer, self).__init__()

    def set_parameters(self, vocab_size, dmodel, output_size, max_len, padding_idx=0, n_layers=4,
                       ffnn_hidden_size=None, heads=8, pooling='max', dropout=0.2):

        if not ffnn_hidden_size:
            ffnn_hidden_size = dmodel * 4

        assert pooling == 'max' or pooling == 'avg', 'Improper pooling type was passed.'

        self.pooling = pooling
        self.output_size = output_size

        self.embedding = nn.Embedding(vocab_size, dmodel).to(device)
        self.pos_encoding = PositionalEncoding(max_len, dmodel, dropout, padding_idx).to(device)

        self.tnf_blocks = nn.ModuleList()

        for n in range(n_layers):
            self.tnf_blocks.append(TransformerBlock(dmodel, ffnn_hidden_size, heads, dropout).to(device))

        self.tnf_blocks = nn.Sequential(*self.tnf_blocks)

        self.linear = nn.Linear(dmodel, output_size)

    def forward(self, inputs, input_lengths):
        """Forward propagate through the Transformer.

        Parameters
        ----------
        inputs: torch.Tensor
            Batch of input sequences.
        input_lengths: torch.LongTensor
            Batch containing sequences lengths.

        Returns
        -------
        torch.Tensor
            Logarithm of softmaxed class tensor.
        """
        self.batch_size = inputs.size(0)

        # Input dimensions (batch_size, seq_length, dmodel)
        output = self.embedding(inputs)
        output = self.pos_encoding(output)
        output = self.tnf_blocks(output)
        # Output dimensions (batch_size, seq_length, dmodel)

        if self.pooling == 'max':
            # Permute to the shape (batch_size, dmodel, seq_length)
            # Apply max-pooling, output dimensions (batch_size, dmodel)
            output = F.adaptive_max_pool1d(output.permute(0, 2, 1), (1,)).view(self.batch_size, -1)
        else:
            # Sum along the batch axis and divide by the corresponding lengths (FloatTensor)
            # Output shape: (batch_size, dmodel)
            output = torch.sum(output, dim=1) / input_lengths.view(-1, 1).type(torch.FloatTensor)

        output = self.linear(output)

        return F.log_softmax(output, dim=-1)

    def add_loss_fn(self, loss_fn):
        """Add loss function to the model.

        """
        self.loss_fn = loss_fn

    def add_optimizer(self, optimizer):
        """Add optimizer to the model.

        """
        self.optimizer = optimizer

    def add_device(self, device=torch.device('cpu')):
        """Specify the device.

        """
        self.device = device

    def train_model(self, train_iterator):
        """Perform single training epoch.

        Parameters
        ----------
        train_iterator: BatchIterator
            BatchIterator class object containing training batches.

        Returns
        -------
        train_losses: list
            List of the training average batch losses.
        avg_loss: float
            Average loss on the entire training set.
        accuracy: float
            Models accuracy on the entire training set.

        """
        self.train()

        train_losses = []
        losses = []
        losses_list = []
        num_seq = 0
        batch_correct = 0

        for i, batches in tqdm.tqdm_notebook(enumerate(train_iterator, 1), total=len(train_iterator), desc='Training'):
            input_seq, target, x_lengths = batches['input_seq'], batches['target'], batches['x_lengths']

            # input_seq.to(self.device)
            # target.to(self.device)
            # x_lengths.to(self.device)
            input_seq = input_seq.to(device)
            target = target.to(device)
            x_lengths = x_lengths.to(device)

            self.optimizer.zero_grad()

            pred = self.forward(input_seq, x_lengths)
            loss = self.loss_fn(pred, target)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()

            losses_list.append(loss.data.cpu().numpy())

            pred = torch.argmax(pred, 1)

            if self.device.type == 'cpu':
                batch_correct += (pred.cpu() == target.cpu()).sum().item()

            else:
                batch_correct += (pred == target).sum().item()

            num_seq += len(input_seq)


            if i % 100 == 0:
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)

                accuracy = batch_correct / num_seq

                print('Iteration: {}. Average training loss: {:.4f}. Accuracy: {:.3f}'.format(i, avg_train_loss,
                                                                                              accuracy))

                losses = []

            avg_loss = np.mean(losses_list)
            accuracy = batch_correct / num_seq

        return train_losses, avg_loss, accuracy

    def evaluate_model(self, eval_iterator, conf_mtx=False):
        """Perform the one evaluation epoch.

        Parameters
        ----------
        eval_iterator: BatchIterator
            BatchIterator class object containing evaluation batches.
        conf_mtx: boolean, optional (default=False)
            Whether to print the confusion matrix at each epoch.

        Returns
        -------
        eval_losses: list
            List of the evaluation average batch losses.
        avg_loss: float
            Average loss on the entire evaluation set.
        accuracy: float
            Models accuracy on the entire evaluation set.
        conf_matrix: list
            Confusion matrix.

        """
        self.eval()

        eval_losses = []
        losses = []
        losses_list = []
        num_seq = 0
        batch_correct = 0
        pred_total = torch.LongTensor()
        target_total = torch.LongTensor()

        with torch.no_grad():
            for i, batches in tqdm.tqdm_notebook(enumerate(eval_iterator, 1), total=len(eval_iterator), desc='Evaluation'):
                input_seq, target, x_lengths = batches['input_seq'], batches['target'], batches['x_lengths']

                # input_seq.to(self.device)
                # target.to(self.device)
                # x_lengths.to(self.device)
                input_seq = input_seq.to(self.device)
                target = target.to(self.device)
                x_lengths = x_lengths.to(self.device)

                pred = self.forward(input_seq, x_lengths)
                loss = self.loss_fn(pred, target)
                losses.append(loss.data.cpu().numpy())
                losses_list.append(loss.data.cpu().numpy())

                pred = torch.argmax(pred, 1)

                if self.device.type == 'cpu':
                    batch_correct += (pred.cpu() == target.cpu()).sum().item()

                else:
                    batch_correct += (pred == target).sum().item()

                num_seq += len(input_seq)
                pred_total = torch.cat([pred_total.to(device), pred], dim=0)
                target_total = torch.cat([target_total.to(device), target], dim=0)

                if i % 100 == 0:
                    avg_batch_eval_loss = np.mean(losses)
                    eval_losses.append(avg_batch_eval_loss)

                    accuracy = batch_correct / num_seq

                    print('Iteration: {}. Average evaluation loss: {:.4f}. Accuracy: {:.2f}'.format(i,
                                                                                                    avg_batch_eval_loss,
                                                                                                    accuracy))

                    losses = []

            avg_loss_list = []

            avg_loss = np.mean(losses_list)
            accuracy = batch_correct / num_seq

            conf_matrix = confusion_matrix(target_total.to('cpu').view(-1), pred_total.to('cpu').view(-1))

        if conf_mtx:
            print('\tConfusion matrix: ', conf_matrix)

        return eval_losses, avg_loss, accuracy, conf_matrix


model = Transformer()

if CUDA:
    print("CUDA is available")
    # model.cuda()
    model.to(device)

model.add_device(device)

train_dataset = pd.read_csv('dataset/datasets_feat_clean/train_feat_clean.csv',
                            usecols=['clean_review', 'label'])
train_dataset = train_dataset[['clean_review', 'label']]
train_dataset.head()

val_dataset = pd.read_csv('dataset/datasets_feat_clean/val_feat_clean.csv',
                          usecols=['clean_review', 'label'])
val_dataset = val_dataset[['clean_review', 'label']]
val_dataset.head()

batch_size = 32

train_iterator = BatchIterator(train_dataset, batch_size=batch_size, vocab_created=False, vocab=None, target_col=None,
                               word2index=None, sos_token='<SOS>', eos_token='<EOS>', unk_token='<UNK>',
                               pad_token='<PAD>', min_word_count=3, max_vocab_size=None, max_seq_len=0.9,
                               use_pretrained_vectors=False, glove_path='glove/', glove_name='glove.6B.100d.txt',
                               weights_file_name='glove/weights.npy')

val_iterator = BatchIterator(val_dataset, batch_size=batch_size, vocab_created=False, vocab=None, target_col=None,
                             word2index=train_iterator.word2index, sos_token='<SOS>', eos_token='<EOS>',
                             unk_token='<UNK>', pad_token='<PAD>', min_word_count=3, max_vocab_size=None,
                             max_seq_len=0.9, use_pretrained_vectors=False, glove_path='glove/',
                             glove_name='glove.6B.100d.txt', weights_file_name='glove/weights.npy')

for batches in train_iterator:
    input_seq, target, x_lengths = batches['input_seq'], batches['target'], batches['x_lengths']
    print('input_seq shape: ', input_seq.size())
    print('target shape: ', target.size())
    print('x_lengths shape: ', x_lengths.size())
    break

max_len = 0

for batches in train_iterator:
    x_lengths = batches['x_lengths']
    if max(x_lengths) > max_len:
        max_len = int(max(x_lengths))

print('Maximum sequence length: {}'.format(max_len))

vocab_size = len(train_iterator.word2index)
dmodel = 64
output_size = 2
padding_idx = train_iterator.word2index['<PAD>']
n_layers = 4
ffnn_hidden_size = dmodel * 2
heads = 8
pooling = 'max'
dropout = 0.5
label_smoothing = 0.1
learning_rate = 0.001
epochs = 30

model.set_parameters(vocab_size, dmodel, output_size, max_len, padding_idx, n_layers, ffnn_hidden_size, heads, pooling,
                     dropout)

if label_smoothing:
    loss_fn = LabelSmoothingLoss(output_size, label_smoothing).to(device)
else:
    loss_fn = nn.NLLLoss()

model.add_loss_fn(loss_fn)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.add_optimizer(optimizer)

params = {'batch_size': batch_size,
          'dmodel': dmodel,
          'n_layers': n_layers,
          'ffnn_hidden_size': ffnn_hidden_size,
          'heads': heads,
          'pooling': pooling,
          'dropout': dropout,
          'label_smoothing': label_smoothing,
          'learning_rate': learning_rate}

train_writer = SummaryWriter(
    comment=f' Training, batch_size={batch_size}, dmodel={dmodel}, n_layers={n_layers},ffnn_hidden_size={ffnn_hidden_size}, heads={heads}, pooling={pooling}, dropout={dropout}, label_smoothing={label_smoothing}, learning_rate={learning_rate}'.format(
        **params))

val_writer = SummaryWriter(
    comment=f' Validation, batch_size={batch_size}, dmodel={dmodel}, n_layers={n_layers},ffnn_hidden_size={ffnn_hidden_size}, heads={heads}, pooling={pooling}, dropout={dropout}, label_smoothing={label_smoothing}, learning_rate={learning_rate}'.format(
        **params))

early_stop = EarlyStopping(wait_epochs=3)

train_losses_list, train_avg_loss_list, train_accuracy_list = [], [], []
eval_avg_loss_list, eval_accuracy_list, conf_matrix_list = [], [], []

for epoch in range(epochs):

    try:
        print('\nStart epoch [{}/{}]'.format(epoch + 1, epochs))
        model.to(device)
        train_losses, train_avg_loss, train_accuracy = model.train_model(train_iterator)

        train_losses_list.append(train_losses)
        train_avg_loss_list.append(train_avg_loss)
        train_accuracy_list.append(train_accuracy)

        _, eval_avg_loss, eval_accuracy, conf_matrix = model.evaluate_model(val_iterator)

        eval_avg_loss_list.append(eval_avg_loss)
        eval_accuracy_list.append(eval_accuracy)
        conf_matrix_list.append(conf_matrix)

        print(
            '\nEpoch [{}/{}]: Train accuracy: {:.3f}. Train loss: {:.4f}. Evaluation accuracy: {:.3f}. Evaluation loss: {:.4f}'.format(
                epoch + 1, epochs, train_accuracy, train_avg_loss, eval_accuracy, eval_avg_loss))

        train_writer.add_scalar('Training loss', train_avg_loss, epoch)
        val_writer.add_scalar('Validation loss', eval_avg_loss, epoch)

        if early_stop.stop(eval_avg_loss, model, delta=0.003):
            break

    finally:
        train_writer.close()
        val_writer.close()

test_dataset = pd.read_csv('dataset/datasets_feat_clean/test_feat_clean.csv',
                           usecols=['clean_review', 'label'])
test_dataset = test_dataset[['clean_review', 'label']]
test_dataset.head()

test_iterator = BatchIterator(test_dataset, batch_size=256, vocab_created=False, vocab=None, target_col=None,
                              word2index=train_iterator.word2index, sos_token='<SOS>', eos_token='<EOS>',
                              unk_token='<UNK>', pad_token='<PAD>', min_word_count=3, max_vocab_size=None,
                              max_seq_len=0.9, use_pretrained_vectors=False, glove_path='glove/',
                              glove_name='glove.6B.100d.txt', weights_file_name='glove/weights.npy')

_, test_avg_loss, test_accuracy, test_conf_matrix = model.evaluate_model(test_iterator)
print('Test accuracy: {:.3f}. Test error: {:.3f}'.format(test_accuracy, test_avg_loss))
