import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from .set_encoder import SetEncoder
from .beam_search import BeamHypotheses
import numpy as np
from tqdm import tqdm 
from ..dataset.generator import Generator, InvalidPrefixExpression
from itertools import chain
from sympy import lambdify 
from . import bfgs
from concurrent.futures import ProcessPoolExecutor, as_completed


class Model(pl.LightningModule):
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        self.trg_pad_idx = cfg.trg_pad_idx
        self.tok_embedding = nn.Embedding(cfg.output_dim, cfg.dim_hidden)
        self.pos_embedding = nn.Embedding(cfg.length_eq, cfg.dim_hidden)
        if cfg.sinuisodal_embeddings:
            self.create_sinusoidal_embeddings(
                cfg.length_eq, cfg.dim_hidden, out=self.pos_embedding.weight
            )
        # -=+=-=+=-=+=- Model START -=+=-=+=-=+=-
        self.enc = SetEncoder(cfg)
        decoder_layer = nn.TransformerDecoderLayer(  # One Layer
            d_model=cfg.dim_hidden,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dec_pf_dim,
            dropout=cfg.dropout,
        )
        self.decoder_transfomer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.dec_layers) # Total decoder

        self.fc_out = nn.Linear(cfg.dim_hidden, cfg.output_dim)
        # -=+=-=+=-=+=- Model END -=+=-=+=-=+=-

        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.dropout = nn.Dropout(cfg.dropout)
        self.eq = None
        
        # TO BE REMOVED
        self.counter = 0

    def create_sinusoidal_embeddings(self, n_pos, dim, out):
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        out.requires_grad = False

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).float()
        trg_pad_mask = (
            trg_pad_mask.masked_fill(trg_pad_mask == 0, float("-inf"))
            .masked_fill(trg_pad_mask == 1, float(0.0))
            .type_as(trg)
        )
        trg_len = trg.shape[1]
        mask = (torch.triu(torch.ones(trg_len, trg_len)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .type_as(trg)
        )
        return trg_pad_mask, mask

    def forward(self,batch):
        # print(batch[0].shape)
        # print(batch[1].shape)
        if self.counter > 5:
            exit()
        b = batch[0].permute(0, 2, 1)
        size = b.shape[-1]
        src_x = b[:, :, : (size - 1)]
        src_y = b[:, :, -1].unsqueeze(2)
        trg = batch[1].long()
        trg_mask1, trg_mask2 = self.make_trg_mask(trg[:, :-1])
        src_mask = None
        encoder_input = torch.cat((src_x, src_y), dim=-1)

        enc_src = self.enc(encoder_input)

        assert not torch.isnan(enc_src).any()
        pos = self.pos_embedding(
            torch.arange(0, batch[1].shape[1] - 1)
            .unsqueeze(0)
            .repeat(batch[1].shape[0], 1)
            .type_as(trg)
        )
        te = self.tok_embedding(trg[:, :-1])
        trg_ = self.dropout(te + pos)
        output = self.decoder_transfomer(
            trg_.permute(1, 0, 2),
            enc_src.permute(1, 0, 2),
            trg_mask2.bool(),
            tgt_key_padding_mask=trg_mask1.bool(),
        ) 
        output = self.fc_out(output)
        # print(output.shape, trg.shape)
        # print(trg[0, :])
        # exit()
        id2word = {'4': 'x_1', '5': 'x_2', '6': 'x_3', '7': 'abs', '8': 'acos', '9': 'add', '10': 'asin', '11': 'atan', '12': 'cos', '13': 'cosh', '14': 'coth', '15': 'div', '16': 'exp', '17': 'ln', '18': 'mul', '19': 'pow', '20': 'sin', '21': 'sinh', '22': 'sqrt', '23': 'tan', '24': 'tanh', '25': '-3', '26': '-2', '27': '-1', '28': '0', '29': '1', '30': '2', '31': '3', '32': '4', '33': '5', '1': '<S>', '2': '<F>', '3': 'c', '0': '<PAD>'}
        # for (target, decode) in zip(trg.tolist(), torch.argmax(output, dim=-1).tolist()):
        #     print([id2word[str(id)] for id in target])
        #     print([id2word[str(id)] for id in decode])
        #     print()
        #     break
        
        # for (target, decode) in zip(trg.tolist(), output):
        #     argmaxdecode = torch.argmax(decode, dim=-1).tolist()
        #     decode_values = decode.tolist()
        #     target_words = [id2word[str(id)] for id in target]
        #     predicted_words = [id2word[str(id)] for id in argmaxdecode]
            
        #     decode_ranks = np.argsort(decode_values, axis=0)
        #     rank_correct = 

        #     print("Target words:", target_words)
        #     print("Predicted words:", predicted_words)

        #     break
        
        for (target, decode) in zip(trg.tolist(), output):
            argmaxdecode = torch.argmax(decode, dim=-1).tolist()  # Predicted labels
            decode_values = decode.tolist()  # Values for each label
            target_words = [id2word[str(id)] for id in target]  # Target words
            predicted_words = [id2word[str(id)] for id in argmaxdecode]  # Predicted words
            
            print("Target tokens:", target_words)
            print("Predicted tokens:", predicted_words)

            # Iterate through each target word and predicted values
            for i, t_id in enumerate(target[1:]):
                if i >= len(decode_values):
                    break
                
                correct_value = decode_values[i][t_id]  # Value of the correct target label
                decode_ranks = np.argsort(decode_values[i])[::-1]  # Descending sort
                rank_correct = np.where(decode_ranks == t_id)[0][0] + 1  # Find 1-indexed rank of correct word

                # Get predicted token and its value
                predicted_token_id = argmaxdecode[i]
                predicted_value = decode_values[i][predicted_token_id]

                print(f"Correct: '{id2word[str(t_id)]}' is ranked {rank_correct} with value {correct_value}")
                print(f"Predicted:'{id2word[str(predicted_token_id)]}' has value {predicted_value}")
            
            print()
            break

        self.counter += 1
        # exit()
        return output, trg

    def compute_loss(self, output, trg):
        output = output.permute(1, 0, 2).contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)
        loss = self.criterion(output, trg)
        print(loss)
        return loss

    def training_step(self, batch, _):
        output, trg = self.forward(batch)
        loss = self.compute_loss(output,trg)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        output, trg = self.forward(batch)
        loss = self.compute_loss(output,trg)
        self.log("val_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer

    def fitfunc(self, X,y, cfg_params=None, return_encoder_value=False):
        """Same API as fit functions in sklearn: 
            X [Number_of_points, Number_of_features], 
            Y [Number_of_points]
        """
        X = X
        y = y[:,None]

        # reshaping and padding
        X = X.to(self.device).unsqueeze(0)
        if X.shape[2] < self.cfg.dim_input - 1:
            pad = torch.zeros(1, X.shape[1],self.cfg.dim_input-X.shape[2]-1, device=self.device)
            X = torch.cat((X,pad),dim=2)
        y = y.to(self.device).unsqueeze(0)
        with torch.no_grad():

            # encoder computation
            encoder_input = torch.cat((X, y), dim=2) #.permute(0, 2, 1)
            enc_src = self.enc(encoder_input)
            src_enc = enc_src
            shape_enc_src = (cfg_params.beam_size,) + src_enc.shape[1:]
            enc_src = src_enc.unsqueeze(1).expand((1, cfg_params.beam_size) + src_enc.shape[1:]).contiguous().view(shape_enc_src)
            if return_encoder_value:
                return enc_src

            assert enc_src.size(0) == cfg_params.beam_size

            # start generation with BeamSearch
            generated = torch.zeros(
                [cfg_params.beam_size, self.cfg.length_eq],
                dtype=torch.long,
                device=self.device,
            )
            generated[:, 0] = 1
            cache = {"slen": 0}
            generated_hyps = BeamHypotheses(cfg_params.beam_size, self.cfg.length_eq, 1.0, 1)
            done = False

            # Beam Scores
            beam_scores = torch.zeros(cfg_params.beam_size, device=self.device, dtype=torch.long)
            beam_scores[1:] = -1e9
            cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)
            while cur_len < self.cfg.length_eq:
                generated_mask1, generated_mask2 = self.make_trg_mask(
                    generated[:, :cur_len]
                )

                # positional encodings
                pos = self.pos_embedding(
                    torch.arange(0, cur_len)  #### attention here
                    .unsqueeze(0)
                    .repeat(generated.shape[0], 1)
                    .type_as(generated)
                )
                te = self.tok_embedding(generated[:, :cur_len])
                trg_ = self.dropout(te + pos)

                # Decoder computation
                output = self.decoder_transfomer(
                    trg_.permute(1, 0, 2),
                    enc_src.permute(1, 0, 2),
                    generated_mask2.float(),
                    tgt_key_padding_mask=generated_mask1.bool(),
                )
                # final layer, best scores
                output = self.fc_out(output)
                output = output.permute(1, 0, 2).contiguous()
                scores = F.log_softmax(output[:, -1:, :], dim=-1).squeeze(1)
                assert output[:, -1:, :].shape == (cfg_params.beam_size,1,self.cfg.length_eq,)

                n_words = scores.shape[-1]
                # select next words with scores
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
                _scores = _scores.view(cfg_params.beam_size * n_words)  # (bs, beam_size * n_words)

                next_scores, next_words = torch.topk(_scores, 2 * cfg_params.beam_size, dim=0, largest=True, sorted=True)
                assert len(next_scores) == len(next_words) == 2 * cfg_params.beam_size
                done = done or generated_hyps.is_done(next_scores.max().item())
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words, next_scores):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if (
                        word_id == cfg_params.word2id["F"]
                        or cur_len + 1 == self.cfg.length_eq
                    ):
                        generated_hyps.add(
                            generated[beam_id, :cur_len,].clone().cpu(),
                            value.item(),
                        )
                        # print(generated_hyps.hyp)
                    else:
                        next_sent_beam.append(
                            (value, word_id, beam_id)
                        )

                    # the beam for next step is full
                    if len(next_sent_beam) == cfg_params.beam_size:
                        break

                # update next beam content
                assert (
                    len(next_sent_beam) == 0
                    if cur_len + 1 == self.cfg.length_eq
                    else cfg_params.beam_size
                )
                if len(next_sent_beam) == 0:
                    next_sent_beam = [
                        (0, self.trg_pad_idx, 0)
                    ] * cfg_params.beam_size  # pad the batch


                #next_batch_beam.extend(next_sent_beam)
                assert len(next_sent_beam) == cfg_params.beam_size

                beam_scores = torch.tensor(
                    [x[0] for x in next_sent_beam], device=self.device
                )  # .type(torch.int64) Maybe #beam_scores.new_tensor([x[0] for x in next_batch_beam])
                beam_words = torch.tensor(
                    [x[1] for x in next_sent_beam], device=self.device
                )  # generated.new([x[1] for x in next_batch_beam])
                beam_idx = torch.tensor(
                    [x[2] for x in next_sent_beam], device=self.device
                )
                generated = generated[beam_idx, :]
                generated[:, cur_len] = beam_words
                for k in cache.keys():
                    if k != "slen":
                        cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])
                # update current length
                cur_len = cur_len + torch.tensor(
                    1, device=self.device, dtype=torch.int64
                )


            #perc = 0
            #cnt = 0
            #gts = []
            best_preds = []
            best_preds_bfgs = []
            #best_L = []
            best_L_bfgs = []

            #flag = 0
            L_bfgs = []
            P_bfgs = []
            #counter = 1

            #fun_args = ",".join(chain(cfg_params.total_variables,"constant"))
            cfg_params.id2word[3] = "constant"
            for __, ww in sorted(
                generated_hyps.hyp, key=lambda x: x[0], reverse=True
            ):
                try:
                    pred_w_c, constants, loss_bfgs, exa = bfgs.bfgs(
                        ww, X, y, cfg_params
                    )
                except InvalidPrefixExpression:
                    continue
                #L_bfgs = loss_bfgs
                P_bfgs.append(str(pred_w_c))
                L_bfgs.append(loss_bfgs)

            if all(np.isnan(np.array(L_bfgs))):
                print("Warning all nans")
                L_bfgs = float("nan")
                best_L_bfgs = None
            else:
                best_preds_bfgs.append(P_bfgs[np.nanargmin(L_bfgs)])
                best_L_bfgs.append(np.nanmin(L_bfgs))

            output = {'all_bfgs_preds':P_bfgs, 'all_bfgs_loss':L_bfgs, 'best_bfgs_preds':best_preds_bfgs, 'best_bfgs_loss':best_L_bfgs}
            self.eq = output['best_bfgs_preds']
            return output

    def get_equation(self,):
        return self.eq


class EncoderOnly(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.enc = SetEncoder(cfg.architecture)
        self.cfg = cfg.architecture
        self.cfg_complete = cfg
        
        self.trg_pad_idx = self.cfg.trg_pad_idx
        self.tok_embedding = nn.Embedding(self.cfg.output_dim, self.cfg.dim_hidden)
        self.pos_embedding = nn.Embedding(self.cfg.length_eq, self.cfg.dim_hidden)
        if self.cfg.sinuisodal_embeddings:
            self.create_sinusoidal_embeddings(
                self.cfg.length_eq, self.cfg.dim_hidden, out=self.pos_embedding.weight
            )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.cfg.dim_hidden,
            nhead=self.cfg.num_heads,
            dim_feedforward=self.cfg.dec_pf_dim,
            dropout=self.cfg.dropout,
        )
        self.decoder_transfomer = nn.TransformerDecoder(decoder_layer, num_layers=self.cfg.dec_layers)

        self.fc_out = nn.Linear(self.cfg.dim_hidden, self.cfg.output_dim)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.eq = None

    def forward(self, X, y):
        "NOTE this is ONLY the encoder because NNsight"
        X = X
        y = y[:, None]

        # reshaping and padding
        X = X.to(self.device).unsqueeze(0)
        if X.shape[2] < self.cfg.dim_input - 1:
            pad = torch.zeros(1, X.shape[1], self.cfg.dim_input - X.shape[2] - 1, device=self.device)
            X = torch.cat((X, pad), dim=2)
        y = y.to(self.device).unsqueeze(0)

        # encoder computation
        with torch.no_grad():
            encoder_input = torch.cat((X, y), dim=2)
            enc_src = self.enc(encoder_input)
            src_enc = enc_src
            # shape_enc_src = (self.cfg_complete.inference.beam_size,) + src_enc.shape[1:]
            # enc_src = src_enc.unsqueeze(1).expand((1, self.cfg_complete.inference.beam_size) + src_enc.shape[1:]).contiguous().view(shape_enc_src)
        return src_enc

    def full_forward(self, batch, src_enc=None):
        "ENTIRE FORWARD RUN"
        b = batch[0].permute(0, 2, 1)
        size = b.shape[-1]
        src_x = b[:, :, : (size - 1)]
        src_y = b[:, :, -1].unsqueeze(2)
        trg = batch[1].long()
        trg_mask1, trg_mask2 = self.make_trg_mask(trg[:, :-1])
        src_mask = None
        encoder_input = torch.cat((src_x, src_y), dim=-1)

        if src_enc is not None:
            enc_src = src_enc
        else:
            enc_src = self.enc(encoder_input)

        assert not torch.isnan(enc_src).any()
        pos = self.pos_embedding(
            torch.arange(0, batch[1].shape[1] - 1)
            .unsqueeze(0)
            .repeat(batch[1].shape[0], 1)
            .type_as(trg)
        )
        te = self.tok_embedding(trg[:, :-1])
        trg_ = self.dropout(te + pos)
        output = self.decoder_transfomer(
            trg_.permute(1, 0, 2),
            enc_src.permute(1, 0, 2),
            trg_mask2.bool(),
            tgt_key_padding_mask=trg_mask1.bool(),
        )
        output = self.fc_out(output)
        return output, trg

    def decode_one_step(self, encoder, input_tokens):
        """
        Perform one decoding step given the encoder output and previous input tokens.

        Args:
            encoder: Output of the encoder (enc_src).
            input_tokens: Previous input tokens **list** (seq_len, ).

        Returns:
            scores: Logits for the next token (torch.Tensor of shape [batch_size, vocab_size]).
        """
        with torch.no_grad():
            input_tokens = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).cuda()
            seq_len = len(input_tokens)
            pos = self.pos_embedding(
                torch.arange(0, seq_len)
                .unsqueeze(0)
                # .repeat(batch_size, 1)
                .type_as(input_tokens)
            )

            te = self.tok_embedding(input_tokens)
            trg_ = self.dropout(te + pos)

            trg_mask1, trg_mask2 = self.make_trg_mask(input_tokens)

            decoder_output = self.decoder_transfomer(
                trg_.permute(1, 0, 2),              # [seq_len, batch_size, embedding_dim]
                encoder.permute(1, 0, 2),           # [src_len, batch_size, embedding_dim]
                trg_mask2.bool(),
                tgt_key_padding_mask=trg_mask1.bool(),
            )
            scores = self.fc_out(decoder_output[-1])[0]  # Only take the last step's output
        return scores

    def compute_loss(self, output, trg):
        output = output.permute(1, 0, 2).contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)
        loss = self.criterion(output, trg)
        return loss

    def fitfunc(self, X,y, enc_input=None, cfg_params=None, max_len=20, return_skeleton=False):
        if enc_input is None:
                enc_input = self.forward(X, y)
        X = X
        y = y[:,None]

        # reshaping and padding
        X = X.to(self.device).unsqueeze(0)
        if X.shape[2] < self.cfg.dim_input - 1:
            pad = torch.zeros(1, X.shape[1],self.cfg.dim_input-X.shape[2]-1, device=self.device)
            X = torch.cat((X,pad),dim=2)
        y = y.to(self.device).unsqueeze(0)
        with torch.no_grad():
            
            src_enc = enc_input
            shape_enc_src = (cfg_params.beam_size,) + src_enc.shape[1:]
            enc_src = src_enc.unsqueeze(1).expand((1, cfg_params.beam_size) + src_enc.shape[1:]).contiguous().view(shape_enc_src)

            assert enc_src.size(0) == cfg_params.beam_size

            # start generation with BeamSearch
            generated = torch.zeros(
                [cfg_params.beam_size, max_len],
                dtype=torch.long,
                device=self.device,
            )
            generated[:, 0] = 1
            generated_history = []
            # cache = {"slen": 0}
            generated_hyps = BeamHypotheses(cfg_params.beam_size, max_len, 1.0, 1)
            #print(f"encoder only fitfunc", )
            #print(f"encoder only fitfunc generated_hyps", generated_hyps.hyp)
            #print(max_len)
            done = False

            # Beam Scores
            beam_scores = torch.zeros(cfg_params.beam_size, device=self.device, dtype=torch.long)
            beam_scores[1:] = -1e9
            cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)
            while cur_len < max_len:
                generated_mask1, generated_mask2 = self.make_trg_mask(
                    generated[:, :cur_len]
                )

                # positional encodings
                pos = self.pos_embedding(
                    torch.arange(0, cur_len)
                    .unsqueeze(0)
                    .repeat(generated.shape[0], 1)
                    .type_as(generated)
                )
                te = self.tok_embedding(generated[:, :cur_len])
                trg_ = self.dropout(te + pos)

                # Decoder computation
                output = self.decoder_transfomer(
                    trg_.permute(1, 0, 2),
                    enc_src.permute(1, 0, 2),
                    generated_mask2.float(),
                    tgt_key_padding_mask=generated_mask1.bool(),
                )
                # final layer, best scores
                # print(f"encoder only output", output.shape)

                output = self.fc_out(output)
                output = output.permute(1, 0, 2).contiguous()
                #print(f"encoder only output", output.shape) # beamsize, sequence length, output chars (60)
                
                scores = F.log_softmax(output[:, -1, :], dim=-1)
                #print(f"encoder only scores", scores.shape) # beamsize, output chars (60) (taking last token)
                
                assert output[:, -1:, :].shape == (cfg_params.beam_size, 1, 60)

                n_words = scores.shape[-1]
                # select next words with scores
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
                _scores = _scores.view(cfg_params.beam_size * n_words)  # (bs, beam_size * n_words)
                #print(f"encoder only _scores", _scores.shape)

                next_scores, next_words = torch.topk(_scores, 2 * cfg_params.beam_size, dim=0, largest=True, sorted=True)
                #print(f"encoder only next_scores", next_scores.shape) # 2 * beam size
                #print(f"encoder only next_scores", next_words)
                assert len(next_scores) == len(next_words) == 2 * cfg_params.beam_size
                done = done or generated_hyps.is_done(next_scores.max().item())
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words, next_scores):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words
                    
                    #print(f"encoder only beam_id", beam_id)
                    #print(f"encoder only word_id", word_id)

                    # end of sentence, or next word
                    if (
                        word_id == cfg_params.word2id["F"]
                        or cur_len + 1 == max_len
                    ):
                        generated_hyps.add(
                            generated[beam_id, :cur_len,].clone().cpu(),
                            value.item(),
                        )
                        #print("done")
                    else:
                        next_sent_beam.append(
                            (value, word_id, beam_id)
                        )

                    # the beam for next step is full
                    # this can happen because we do 2x beamsize and we only want beamsize
                    if len(next_sent_beam) == cfg_params.beam_size:
                        break

                # update next beam content
                # should be 0 if we have reached max_len else beam_size
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else cfg_params.beam_size

                if len(next_sent_beam) == 0:
                    next_sent_beam = [
                        (0, self.trg_pad_idx, 0)
                    ] * cfg_params.beam_size  # pad the batch

                assert len(next_sent_beam) == cfg_params.beam_size

                beam_scores = torch.tensor(
                    [x[0] for x in next_sent_beam], device=self.device
                )
                beam_words = torch.tensor(
                    [x[1] for x in next_sent_beam], device=self.device
                )
                beam_idx = torch.tensor(
                    [x[2] for x in next_sent_beam], device=self.device
                )
                generated = generated[beam_idx, :]
                generated[:, cur_len] = beam_words
                generated_history.append(generated.cpu().numpy().tolist())

                # update current length
                cur_len = cur_len + torch.tensor(
                    1, device=self.device, dtype=torch.int64
                )


            best_preds = []
            best_preds_bfgs = []
            best_L_bfgs = []

            L_bfgs = []
            P_bfgs = []

            cfg_params.id2word[3] = "constant"

            sorted_hyps = sorted(generated_hyps.hyp, key=lambda x: x[0], reverse=True)
            P_bfgs = []
            L_bfgs = []
            
            if return_skeleton:
                return sorted_hyps, generated_history

            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(process_ww, ww, X, y, cfg_params): ww for _, ww in sorted_hyps}

                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        pred_w_c, loss_bfgs = result
                        P_bfgs.append(pred_w_c)
                        L_bfgs.append(loss_bfgs)

            if all(np.isnan(np.array(L_bfgs))):
                print("Warning all nans")
                L_bfgs = float("nan")
                best_L_bfgs = None
            else:
                best_preds_bfgs.append(P_bfgs[np.nanargmin(L_bfgs)])
                best_L_bfgs.append(np.nanmin(L_bfgs))

            output = {'all_bfgs_preds':P_bfgs, 'all_bfgs_loss':L_bfgs, 'best_bfgs_preds':best_preds_bfgs, 'best_bfgs_loss':best_L_bfgs}
            self.eq = output['best_bfgs_preds']
            return output, generated_history

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).float()
        trg_pad_mask = (
            trg_pad_mask.masked_fill(trg_pad_mask == 0, float("-inf"))
            .masked_fill(trg_pad_mask == 1, float(0.0))
            .type_as(trg)
        )
        trg_len = trg.shape[1]
        mask = (torch.triu(torch.ones(trg_len, trg_len)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .type_as(trg)
        )
        return trg_pad_mask, mask
    
    def make_trg_mask_decode(self, trg):
        """
        Creates a causal mask to prevent attention to future tokens.
        """
        seq_len = trg.size(1)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=trg.device)).bool()
        return mask
    
def process_ww(ww, X, y, cfg_params):
    try:
        pred_w_c, constants, loss_bfgs, exa = bfgs.bfgs(ww, X, y, cfg_params)
        return str(pred_w_c), loss_bfgs
    except InvalidPrefixExpression:
        return None

if __name__ == "__main__":
        model = SetTransformer(n_l_enc=2,src_pad_idx=0,trg_pad_idx=0,dim_input=6,output_dim=20,dim_hidden=40,dec_layers=1,num_heads=8,dec_pf_dim=40,dec_dropout=0,length_eq=30,lr=
            0.001,num_inds=20,ln=True,num_features=10,is_sin_emb=False, bit32=True,norm=False,activation='linear',linear=False,mean=torch.Tensor([1.]),std=torch.Tensor([1.]),input_normalization=False)
        src_x = torch.rand([2,5,20])
        src_y = torch.sin(torch.norm(src_x, dim=1)).unsqueeze(1)
        inp_1 = torch.cat([src_x,src_y], dim=1)
        inp_2 = torch.randint(0,13,[2,10])
        batch = (inp_1,inp_2)
        print(model)