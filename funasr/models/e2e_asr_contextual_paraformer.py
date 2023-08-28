import logging
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import numpy as np

import torch

from funasr.losses.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from funasr.layers.abs_normalize import AbsNormalize
from funasr.models.ctc import CTC
from funasr.models.decoder.abs_decoder import AbsDecoder
from funasr.models.encoder.abs_encoder import AbsEncoder
from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.postencoder.abs_postencoder import AbsPostEncoder
from funasr.models.preencoder.abs_preencoder import AbsPreEncoder
from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.modules.add_sos_eos import add_sos_eos
from funasr.modules.nets_utils import make_pad_mask, pad_list
from funasr.modules.nets_utils import th_accuracy
from funasr.torch_utils.device_funcs import force_gatherable
from funasr.models.e2e_asr_paraformer import Paraformer


if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class NeatContextualParaformer(Paraformer):
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        predictor = None,
        predictor_weight: float = 0.0,
        predictor_bias: int = 0,
        sampling_ratio: float = 0.2,
        target_buffer_length: int = -1,
        inner_dim: int = 256, 
        bias_encoder_type: str = 'lstm',
        use_decoder_embedding: bool = False,
        crit_attn_weight: float = 0.0,
        crit_attn_smooth: float = 0.0,
        bias_encoder_dropout_rate: float = 0.0,
        preencoder: Optional[AbsPreEncoder] = None,
        postencoder: Optional[AbsPostEncoder] = None,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__(
        vocab_size=vocab_size,
        token_list=token_list,
        frontend=frontend,
        specaug=specaug,
        normalize=normalize,
        preencoder=preencoder,
        encoder=encoder,
        postencoder=postencoder,
        decoder=decoder,
        ctc=ctc,
        ctc_weight=ctc_weight,
        interctc_weight=interctc_weight,
        ignore_id=ignore_id,
        blank_id=blank_id,
        sos=sos,
        eos=eos,
        lsm_weight=lsm_weight,
        length_normalized_loss=length_normalized_loss,
        report_cer=report_cer,
        report_wer=report_wer,
        sym_space=sym_space,
        sym_blank=sym_blank,
        extract_feats_in_collect_stats=extract_feats_in_collect_stats,
        predictor=predictor,
        predictor_weight=predictor_weight,
        predictor_bias=predictor_bias,
        sampling_ratio=sampling_ratio,
        )

        if bias_encoder_type == 'lstm':
            logging.warning("enable bias encoder sampling and contextual training")
            self.bias_encoder = torch.nn.LSTM(inner_dim, inner_dim, 1, batch_first=True, dropout=bias_encoder_dropout_rate)
            self.bias_embed = torch.nn.Embedding(vocab_size, inner_dim)
        elif bias_encoder_type == 'mean':
            logging.warning("enable bias encoder sampling and contextual training")
            self.bias_embed = torch.nn.Embedding(vocab_size, inner_dim)
        else:
            logging.error("Unsupport bias encoder type: {}".format(bias_encoder_type))

        self.target_buffer_length = target_buffer_length
        if self.target_buffer_length > 0:
            self.hotword_buffer = None
            self.length_record = []
            self.current_buffer_length = 0
        self.use_decoder_embedding = use_decoder_embedding
        self.crit_attn_weight = crit_attn_weight
        if self.crit_attn_weight > 0:
            self.attn_loss = torch.nn.L1Loss()
        self.crit_attn_smooth = crit_attn_smooth

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            hotword_pad: torch.Tensor,
            hotword_lengths: torch.Tensor,
            ideal_attn: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
                speech.shape[0]
                == speech_lengths.shape[0]
                == text.shape[0]
                == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]
        self.step_cur += 1
        # for data-parallel
        text = text[:, : text_lengths.max()]
        speech = speech[:, :speech_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_pre = None
        loss_ideal = None

        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text, text_lengths
                )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (1 - self.interctc_weight) * loss_ctc + self.interctc_weight * loss_interctc

        # 2b. Attention decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att, cer_att, wer_att, loss_pre, loss_ideal = self._calc_att_clas_loss(
                encoder_out, encoder_out_lens, text, text_lengths, hotword_pad, hotword_lengths, ideal_attn
            )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att + loss_pre * self.predictor_weight
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att + loss_pre * self.predictor_weight

        if loss_ideal is not None:
            loss = loss + loss_ideal * self.crit_attn_weight
            stats["loss_ideal"] = loss_ideal.detach().cpu()

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att
        stats["loss_pre"] = loss_pre.detach().cpu() if loss_pre is not None else None

        stats["loss"] = torch.clone(loss.detach())
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight
    
    def _calc_att_clas_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
            hotword_pad: torch.Tensor,
            hotword_lengths: torch.Tensor,
            ideal_attn: torch.Tensor,
    ):
        encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
            encoder_out.device)
        if self.predictor_bias == 1:
            _, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = ys_pad_lens + self.predictor_bias
        pre_acoustic_embeds, pre_token_length, _, _ = self.predictor(encoder_out, ys_pad, encoder_out_mask,
                                                                                  ignore_id=self.ignore_id)

        # -1. bias encoder
        if self.use_decoder_embedding:
            hw_embed = self.decoder.embed(hotword_pad)
        else:
            hw_embed = self.bias_embed(hotword_pad)
        hw_embed, (_, _) = self.bias_encoder(hw_embed)
        _ind = np.arange(0, hotword_pad.shape[0]).tolist()
        selected = hw_embed[_ind, [i-1 for i in hotword_lengths.detach().cpu().tolist()]]
        contextual_info = selected.squeeze(0).repeat(ys_pad.shape[0], 1, 1).to(ys_pad.device)

        # 0. sampler
        decoder_out_1st = None
        if self.sampling_ratio > 0.0:
            if self.step_cur < 2:
                logging.info("enable sampler in paraformer, sampling_ratio: {}".format(self.sampling_ratio))
            sematic_embeds, decoder_out_1st = self.sampler(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens,
                                                           pre_acoustic_embeds, contextual_info)
        else:
            if self.step_cur < 2:
                logging.info("disable sampler in paraformer, sampling_ratio: {}".format(self.sampling_ratio))
            sematic_embeds = pre_acoustic_embeds

        # 1. Forward decoder
        decoder_outs = self.decoder(
            encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens, contextual_info=contextual_info
        ) 
        decoder_out, _ = decoder_outs[0], decoder_outs[1]
        '''
        if self.crit_attn_weight > 0 and attn.shape[-1] > 1:
            ideal_attn = ideal_attn + self.crit_attn_smooth / (self.crit_attn_smooth + 1.0)
            attn_non_blank = attn[:,:,:,:-1]
            ideal_attn_non_blank = ideal_attn[:,:,:-1]
            loss_ideal = self.attn_loss(attn_non_blank.max(1)[0], ideal_attn_non_blank.to(attn.device))
        else:
            loss_ideal = None
        '''
        loss_ideal = None

        if decoder_out_1st is None:
            decoder_out_1st = decoder_out
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(
            decoder_out_1st.view(-1, self.vocab_size),
            ys_pad,
            ignore_label=self.ignore_id,
        )
        loss_pre = self.criterion_pre(ys_pad_lens.type_as(pre_token_length), pre_token_length)

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out_1st.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att, loss_pre, loss_ideal
    
    def sampler(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, pre_acoustic_embeds, contextual_info):

        tgt_mask = (~make_pad_mask(ys_pad_lens, maxlen=ys_pad_lens.max())[:, :, None]).to(ys_pad.device)
        ys_pad = ys_pad * tgt_mask[:, :, 0]
        if self.share_embedding:
            ys_pad_embed = self.decoder.output_layer.weight[ys_pad]
        else:
            ys_pad_embed = self.decoder.embed(ys_pad)
        with torch.no_grad():
            decoder_outs = self.decoder(
                encoder_out, encoder_out_lens, pre_acoustic_embeds, ys_pad_lens, contextual_info=contextual_info
            )
            decoder_out, _ = decoder_outs[0], decoder_outs[1]
            pred_tokens = decoder_out.argmax(-1)
            nonpad_positions = ys_pad.ne(self.ignore_id)
            seq_lens = (nonpad_positions).sum(1)
            same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
            input_mask = torch.ones_like(nonpad_positions)
            bsz, seq_len = ys_pad.size()
            for li in range(bsz):
                target_num = (((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio).long()
                if target_num > 0:
                    input_mask[li].scatter_(dim=0, index=torch.randperm(seq_lens[li])[:target_num].to(pre_acoustic_embeds.device), value=0)
            input_mask = input_mask.eq(1)
            input_mask = input_mask.masked_fill(~nonpad_positions, False)
            input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)

        sematic_embeds = pre_acoustic_embeds.masked_fill(~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
            input_mask_expand_dim, 0)
        return sematic_embeds * tgt_mask, decoder_out * tgt_mask

    def cal_decoder_with_predictor(self, encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens, hw_list=None, clas_scale=1.0):
        if hw_list is None:
            hw_list = [torch.Tensor([1]).long().to(encoder_out.device)]  # empty hotword list
            hw_list_pad = pad_list(hw_list, 0)
            if self.use_decoder_embedding:
                hw_embed = self.decoder.embed(hw_list_pad)
            else:
                hw_embed = self.bias_embed(hw_list_pad)
            hw_embed, (h_n, _) = self.bias_encoder(hw_embed)
        else:
            hw_lengths = [len(i) for i in hw_list]
            hw_list_pad = pad_list([torch.Tensor(i).long() for i in hw_list], 0).to(encoder_out.device)
            if self.use_decoder_embedding:
                hw_embed = self.decoder.embed(hw_list_pad)
            else:
                hw_embed = self.bias_embed(hw_list_pad)
            hw_embed = torch.nn.utils.rnn.pack_padded_sequence(hw_embed, hw_lengths, batch_first=True,
                                                            enforce_sorted=False)
            _, (h_n, _) = self.bias_encoder(hw_embed)
            hw_embed = h_n.repeat(encoder_out.shape[0], 1, 1)
        
        decoder_outs = self.decoder(
            encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens, contextual_info=hw_embed, clas_scale=clas_scale
        )
        decoder_out = decoder_outs[0]
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        return decoder_out, ys_pad_lens


class SeACoParaformer(Paraformer): # decoder hotword augmented paraformer
    """
    
    """
 
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        decoder2: AbsDecoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        predictor = None,
        predictor_weight: float = 0.0,
        predictor_bias: int = 0,
        sampling_ratio: float = 0.2,
        inner_dim: int = 256,
        hw_ctc_weight: float = 0.3,
        bias_decoder_type: str = 'lstm',
        bias_decoder_dropout: float = 0.1,
        ideal_weight: float = 0.0,
        train_decoder: bool = False,
        bias_bd_lstm: bool = False,
        phone_embedding: bool =False,
        no_bias: int = 8377,
        preencoder=None,
        postencoder=None,
    ):
        super().__init__(
        vocab_size=vocab_size,
        token_list=token_list,
        frontend=frontend,
        specaug=specaug,
        normalize=normalize,
        encoder=encoder,
        decoder=decoder,
        ctc=ctc,
        ctc_weight=ctc_weight,
        interctc_weight=interctc_weight,
        ignore_id=ignore_id,
        blank_id=blank_id,
        sos=sos,
        eos=eos,
        lsm_weight=lsm_weight,
        length_normalized_loss=length_normalized_loss,
        report_cer=report_cer,
        report_wer=report_wer,
        sym_space=sym_space,
        sym_blank=sym_blank,
        extract_feats_in_collect_stats=extract_feats_in_collect_stats,
        predictor=predictor,
        predictor_weight=predictor_weight,
        predictor_bias=predictor_bias,
        sampling_ratio=sampling_ratio,
        )
 
        self.inner_dim = inner_dim
        self.hw_ctc_weight = hw_ctc_weight
        self.train_decoder = train_decoder
 
        if bias_decoder_type == 'lstm':
            logging.warning("enable lstm bias decoder sampling and contextual training")
            self.bias_encoder = torch.nn.LSTM(self.inner_dim, 
                                              self.inner_dim, 
                                              1, 
                                              batch_first=True, 
                                              dropout=bias_decoder_dropout,
                                              bidirectional=bias_bd_lstm)
            if bias_bd_lstm:
                self.lstm_proj = torch.nn.Linear(self.inner_dim*2, self.inner_dim)
            else:
                self.lstm_proj = None
        elif bias_decoder_type == 'attn':
            logging.warning("enable attn bias decoder sampling and contextual training")
            from funasr.modules.attention import MultiHeadedAttention
            self.bias_encoder = MultiHeadedAttention(4, self.inner_dim, bias_decoder_dropout)
            self.bias_encoder_linear = torch.nn.Linear(self.inner_dim, self.inner_dim)
        else:
            logging.error("Unsupported bias decoder type.")
        self.bias_decoder_type = bias_decoder_type

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.decoder2 = decoder2
        self.hotword_output_layer = torch.nn.Linear(self.inner_dim, vocab_size)
        self.phone_embedding = phone_embedding
        self.NOBIAS = no_bias
        self._unique_components()    

    def _unique_components(self):
        # not used
        self.cif_prelu = torch.nn.PReLU(num_parameters=1, init=0.25)
        self.dec_prelu = torch.nn.PReLU(num_parameters=1, init=0.25)
    
    def _hotword_representation(self, 
                                hotword_pad, 
                                hotword_lengths, 
                                hotword_phone_pad=None, 
                                hotword_phone_lengths=None,
                                phone_only=False):
        if self.bias_decoder_type == 'lstm':
            hw_embed = self.decoder.embed(hotword_pad)
            hw_embed, (_, _) = self.bias_encoder(hw_embed)
            if self.lstm_proj is not None:
                hw_embed = self.lstm_proj(hw_embed)
            _ind = np.arange(0, hw_embed.shape[0]).tolist()
            selected = hw_embed[_ind, [i-1 for i in hotword_lengths.detach().cpu().tolist()]]
            if hotword_phone_pad is not None:
                hw_phone_embed = self.phone_embed(hotword_phone_pad)
                hw_phone_embed, (_, _) = self.phone_lstm(hw_phone_embed)
                _ind = np.arange(0, hw_phone_embed.shape[0]).tolist()
                phone_selected = hw_phone_embed[_ind, [i-1 for i in hotword_phone_lengths.detach().cpu().tolist()]]
                if phone_only:
                    selected = phone_selected
                else:
                    selected += phone_selected
        elif self.bias_decoder_type == 'attn':
            # padding a blank token in the start
            _pad = torch.zeros(hotword_pad.shape[0], 1).int().to(hotword_pad.device)
            hotword_pad = torch.cat([_pad, hotword_pad], dim=1)
            hw_embed = self.decoder.embed(hotword_pad)
            mask = ~make_pad_mask(hotword_lengths+1, maxlen=hotword_pad.shape[1])[:, None, :].to(hotword_pad.device)
            hotword_attn_res = self.bias_encoder(hw_embed, hw_embed, hw_embed, mask)[:, 0] # N*(L+1)*D
            hotword_attn_linear = self.bias_encoder_linear(hotword_attn_res)
            selected = hotword_attn_res + hotword_attn_linear # N*D
        return selected
 
    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            hotword_pad: torch.Tensor,
            hotword_lengths: torch.Tensor,
            dha_pad: torch.Tensor = None,
            hotword_phone_pad: torch.Tensor = None,
            hotword_phone_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss
 
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
                speech.shape[0]
                == speech_lengths.shape[0]
                == text.shape[0]
                == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]
        self.step_cur += 1
        # for data-parallel
        text = text[:, : text_lengths.max()]
        speech = speech[:, :speech_lengths.max()]
 
        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if self.predictor_bias == 1:
            _, ys_pad = add_sos_eos(text, self.sos, self.eos, self.ignore_id)
            ys_lengths = text_lengths + self.predictor_bias

        stats = dict() 
        if hotword_phone_pad is None:
            loss_dha = self._calc_dha_loss(
                encoder_out, encoder_out_lens, ys_pad, ys_lengths, hotword_pad, hotword_lengths, dha_pad
            )
        else:
            loss_dha = self._calc_dha_loss(encoder_out, 
                                           encoder_out_lens, 
                                           ys_pad, 
                                           ys_lengths, 
                                           hotword_pad, 
                                           hotword_lengths, 
                                           dha_pad,
                                           hotword_phone_pad,
                                           hotword_phone_lengths,
                                           )
        if self.train_decoder:
            loss_att, acc_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
            loss = loss_dha + loss_att
            stats["loss_att"] = torch.clone(loss_att.detach())
            stats["acc_att"] = acc_att
        else:
            loss = loss_dha
        stats["loss_dha"] = torch.clone(loss_dha.detach())
        stats["loss"] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _merge(self, cif_attended, dec_attended):
        return cif_attended + dec_attended
    
    def _calc_dha_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_lengths: torch.Tensor,
            hotword_pad: torch.Tensor,
            hotword_lengths: torch.Tensor,
            dha_pad: torch.Tensor,
            hotword_phone_pad: torch.Tensor = None,
            hotword_phone_lengths: torch.Tensor = None,
    ):  
        # predictor forward
        encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
            encoder_out.device)
        pre_acoustic_embeds, _, _, _ = self.predictor(encoder_out, ys_pad, encoder_out_mask,
                                                                                  ignore_id=self.ignore_id)
        # decoder forward
        decoder_out, _ = self.decoder(encoder_out, encoder_out_lens, pre_acoustic_embeds, ys_lengths, return_hidden=True)
        selected = self._hotword_representation(hotword_pad, 
                                                hotword_lengths,
                                                hotword_phone_pad=hotword_phone_pad,
                                                hotword_phone_lengths=hotword_phone_lengths)
        contextual_info = selected.squeeze(0).repeat(encoder_out.shape[0], 1, 1).to(encoder_out.device)
        num_hot_word = contextual_info.shape[1]
        _contextual_length = torch.Tensor([num_hot_word]).int().repeat(encoder_out.shape[0]).to(encoder_out.device)
        # dha core
        cif_attended, _ = self.decoder2(contextual_info, _contextual_length, pre_acoustic_embeds, ys_lengths)
        dec_attended, _ = self.decoder2(contextual_info, _contextual_length, decoder_out, ys_lengths)
        merged = self._merge(cif_attended, dec_attended)
        dha_output = self.hotword_output_layer(merged[:, :-1])  # remove the last token in loss calculation
        loss_att = self.criterion_att(dha_output, dha_pad)
        return loss_att

    def cal_decoder_with_ASF(self, 
                            encoder_out, 
                            encoder_out_lens, 
                            sematic_embeds, 
                            ys_pad_lens, 
                            hw_list,
                            nfilter=50,
                            lmbd=1.0):
        hw_lengths = [len(i) for i in hw_list]
        hw_list_pad = pad_list([torch.Tensor(i).long() for i in hw_list], 0).to(encoder_out.device)
        # decoder forward
        decoder_out, decoder_hidden, _ = self.decoder(encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens, return_hidden=True, return_both=True)
        decoder_pred = torch.log_softmax(decoder_out, dim=-1)
        if hw_list_pad is None:
            return decoder_pred, ys_pad_lens
        selected = self._hotword_representation(hw_list_pad, torch.Tensor(hw_lengths).int().to(encoder_out.device))
        contextual_info = selected.squeeze(0).repeat(encoder_out.shape[0], 1, 1).to(encoder_out.device)
        num_hot_word = contextual_info.shape[1]
        _contextual_length = torch.Tensor([num_hot_word]).int().repeat(encoder_out.shape[0]).to(encoder_out.device)
        # dha core
        if nfilter: # ASF
            for dec in self.decoder2.decoders:
                dec.reserve_attn = True
            cif_attended, _ = self.decoder2(contextual_info, _contextual_length, sematic_embeds, ys_pad_lens)
            dec_attended, _ = self.decoder2(contextual_info, _contextual_length, decoder_hidden, ys_pad_lens)
            # cif_filter = torch.topk(self.decoder2.decoders[-1].attn_mat[0][0].sum(0).sum(0)[:-1], min(nfilter, num_hot_word-1))[1].tolist()
            dec_filter = torch.topk(self.decoder2.decoders[-1].attn_mat[1][0].sum(0).sum(0)[:-1], min(nfilter, num_hot_word-1))[1].tolist()
            add_filter = dec_filter
            add_filter.append(len(hw_list_pad)-1)
            selected = selected[add_filter]
            contextual_info = selected.squeeze(0).repeat(encoder_out.shape[0], 1, 1).to(encoder_out.device)
            num_hot_word = contextual_info.shape[1]
            _contextual_length = torch.Tensor([num_hot_word]).int().repeat(encoder_out.shape[0]).to(encoder_out.device)
            for dec in self.decoder2.decoders:
                dec.attn_mat = []
                dec.reserve_attn = False
        
        cif_attended, _ = self.decoder2(contextual_info, _contextual_length, sematic_embeds, ys_pad_lens)
        dec_attended, _ = self.decoder2(contextual_info, _contextual_length, decoder_hidden, ys_pad_lens)
        
        merged = self._merge(cif_attended, dec_attended)
        dha_output = self.hotword_output_layer(merged[:, :-1])  # remove the last token in loss calculation
        dha_pred = torch.log_softmax(dha_output, dim=-1)
        
        # merging logits
        dha_ids = dha_pred.max(-1)[-1][0]
        for j in range(len(dha_output[0])):
            if dha_ids[j] != self.NOBIAS:
                if j < len(decoder_pred[0]):
                    decoder_pred[0][j] = dha_output[0][j] * lmbd + decoder_pred[0][j] * (1 - lmbd)
        return decoder_pred, ys_pad_lens