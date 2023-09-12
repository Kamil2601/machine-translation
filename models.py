import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1, bidirectional=False):
        super(EncoderGRU, self).__init__()
        self.D = 2 if bidirectional else 1
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            vocab_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x, hidden=None):
        if hidden == None:
            hidden = self.init_hidden(x.shape[0]).to(x.device)

        one_hot = F.one_hot(x, num_classes=self.vocab_size).float().to(x.device)

        output, hidden = self.gru(one_hot, hidden)

        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(
            self.D * self.gru.num_layers,
            batch_size,
            self.hidden_size,
            dtype=torch.float32,
        )
        
        
class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            vocab_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        if hidden == None:
            hidden = self.init_hidden(x.shape[0]).to(x.device)
        one_hot = F.one_hot(x, num_classes=self.vocab_size).float().to(x.device)
        output, hidden = self.gru(one_hot, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(
            self.gru.num_layers, batch_size, self.hidden_size, dtype=torch.float32
        )
        
        
class Seq2sec(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(Seq2sec, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_input_batch, sos_index = 1, dec_input_batch = None, teacher_forcing = False, out_length = 1):
        encoder_output, encoder_hidden = self.encoder(enc_input_batch)
        batch_size = len(enc_input_batch)

        if teacher_forcing:
            decoder_output, _ = self.decoder(dec_input_batch, encoder_hidden)
            return decoder_output
        else:
            decoder_input = (torch.zeros(batch_size, 1, dtype=torch.int64) + sos_index).to(enc_input_batch.device)
            decoder_output = torch.empty(batch_size, out_length, self.decoder.vocab_size).to(enc_input_batch.device)

            hidden = encoder_hidden

            for i in range(out_length):
                decoder_output_i, hidden = self.decoder(decoder_input, hidden)
                decoder_output[:,i:i+1,:] = decoder_output_i
                decoder_input = torch.argmax(decoder_output_i, dim=-1)

            return decoder_output