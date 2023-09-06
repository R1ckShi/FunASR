# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
import os.path
from pathlib import Path
from typing import List, Union, Tuple

import copy
import math
import torch
import librosa
import numpy as np

from .utils.utils import (CharTokenizer, Hypothesis, ONNXRuntimeError,
                          OrtInferSession, TokenIDConverter, get_logger,
                          read_yaml)
from .utils.postprocess_utils import sentence_postprocess
from .utils.frontend import WavFrontend, SinusoidalPositionEncoderOnline
from .utils.timestamp_utils import time_stamp_lfr6_onnx
from .utils.utils import pad_list, make_pad_mask

logging = get_logger()


class Paraformer():
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """
    def __init__(self, model_dir: Union[str, Path] = None,
                 batch_size: int = 1,
                 device_id: Union[str, int] = "-1",
                 plot_timestamp_to: str = "",
                 quantize: bool = False,
                 intra_op_num_threads: int = 4,
                 cache_dir: str = None
                 ):

        if not Path(model_dir).exists():
            from modelscope.hub.snapshot_download import snapshot_download
            try:
                model_dir = snapshot_download(model_dir, cache_dir=cache_dir)
            except:
                raise "model_dir must be model_name in modelscope or local path downloaded from modelscope, but is {}".format(model_dir)
        
        model_file = os.path.join(model_dir, 'model.onnx')
        if quantize:
            model_file = os.path.join(model_dir, 'model_quant.onnx')
        if not os.path.exists(model_file):
            print(".onnx is not exist, begin to export onnx")
            from funasr.export.export_model import ModelExport
            export_model = ModelExport(
                cache_dir=cache_dir,
                onnx=True,
                device="cpu",
                quant=quantize,
            )
            export_model.export(model_dir)
            
        config_file = os.path.join(model_dir, 'config.yaml')
        cmvn_file = os.path.join(model_dir, 'am.mvn')
        config = read_yaml(config_file)

        self.converter = TokenIDConverter(config['token_list'])
        self.tokenizer = CharTokenizer()
        self.frontend = WavFrontend(
            cmvn_file=cmvn_file,
            **config['frontend_conf']
        )
        self.ort_infer = OrtInferSession(model_file, device_id, intra_op_num_threads=intra_op_num_threads)
        self.batch_size = batch_size
        self.plot_timestamp_to = plot_timestamp_to
        if "predictor_bias" in config['model_conf'].keys():
            self.pred_bias = config['model_conf']['predictor_bias']
        else:
            self.pred_bias = 0

    def __call__(self, wav_content: Union[str, np.ndarray, List[str]], **kwargs) -> List:
        waveform_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
        waveform_nums = len(waveform_list)
        asr_res = []
        for beg_idx in range(0, waveform_nums, self.batch_size):
            
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
            try:
                outputs = self.infer(feats, feats_len)
                am_scores, valid_token_lens = outputs[0], outputs[1]
                if len(outputs) == 4:
                    # for BiCifParaformer Inference
                    us_alphas, us_peaks = outputs[2], outputs[3]
                else:
                    us_alphas, us_peaks = None, None
            except ONNXRuntimeError:
                #logging.warning(traceback.format_exc())
                logging.warning("input wav is silence or noise")
                preds = ['']
            else:
                preds = self.decode(am_scores, valid_token_lens)
                if us_peaks is None:
                    for pred in preds:
                        pred = sentence_postprocess(pred)
                        asr_res.append({'preds': pred})
                else:
                    for pred, us_peaks_ in zip(preds, us_peaks):
                        raw_tokens = pred
                        timestamp, timestamp_raw = time_stamp_lfr6_onnx(us_peaks_, copy.copy(raw_tokens))
                        text_proc, timestamp_proc, _ = sentence_postprocess(raw_tokens, timestamp_raw)
                        # logging.warning(timestamp)
                        if len(self.plot_timestamp_to):
                            self.plot_wave_timestamp(waveform_list[0], timestamp, self.plot_timestamp_to)
                        asr_res.append({'preds': text_proc, 'timestamp': timestamp_proc, "raw_tokens": raw_tokens})
        return asr_res

    def plot_wave_timestamp(self, wav, text_timestamp, dest):
        # TODO: Plot the wav and timestamp results with matplotlib
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rc("font", family='Alibaba PuHuiTi')  # set it to a font that your system supports
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(11, 3.5), dpi=320)
        ax2 = ax1.twinx()
        ax2.set_ylim([0, 2.0])
        # plot waveform
        ax1.set_ylim([-0.3, 0.3])
        time = np.arange(wav.shape[0]) / 16000
        ax1.plot(time, wav/wav.max()*0.3, color='gray', alpha=0.4)
        # plot lines and text
        for (char, start, end) in text_timestamp:
            ax1.vlines(start, -0.3, 0.3, ls='--')
            ax1.vlines(end, -0.3, 0.3, ls='--')
            x_adj = 0.045 if char != '<sil>' else 0.12
            ax1.text((start + end) * 0.5 - x_adj, 0, char)
        # plt.legend()
        plotname = "{}/timestamp.png".format(dest)
        plt.savefig(plotname, bbox_inches='tight')

    def load_data(self,
                  wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List:
        def load_wav(path: str) -> np.ndarray:
            waveform, _ = librosa.load(path, sr=fs)
            return waveform

        if isinstance(wav_content, np.ndarray):
            return [wav_content]

        if isinstance(wav_content, str):
            return [load_wav(wav_content)]

        if isinstance(wav_content, list):
            return [load_wav(path) for path in wav_content]

        raise TypeError(
            f'The type of {wav_content} is not in [str, np.ndarray, list]')

    def extract_feat(self,
                     waveform_list: List[np.ndarray]
                     ) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_len = [], []
        for waveform in waveform_list:
            speech, _ = self.frontend.fbank(waveform)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feats.append(feat)
            feats_len.append(feat_len)

        feats = self.pad_feats(feats, np.max(feats_len))
        feats_len = np.array(feats_len).astype(np.int32)
        return feats, feats_len

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, 'constant', constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def infer(self, feats: np.ndarray,
              feats_len: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_infer([feats, feats_len])
        return outputs

    def decode(self, am_scores: np.ndarray, token_nums: int) -> List[str]:
        return [self.decode_one(am_score, token_num)
                for am_score, token_num in zip(am_scores, token_nums)]

    def decode_one(self,
                   am_score: np.ndarray,
                   valid_token_num: int) -> List[str]:
        yseq = am_score.argmax(axis=-1)
        score = am_score.max(axis=-1)
        score = np.sum(score, axis=-1)

        # pad with mask tokens to ensure compatibility with sos/eos tokens
        # asr_model.sos:1  asr_model.eos:2
        yseq = np.array([1] + yseq.tolist() + [2])
        hyp = Hypothesis(yseq=yseq, score=score)

        # remove sos/eos and get results
        last_pos = -1
        token_int = hyp.yseq[1:last_pos].tolist()

        # remove blank symbol id, which is assumed to be 0
        token_int = list(filter(lambda x: x not in (0, 2), token_int))

        # Change integer-ids to tokens
        token = self.converter.ids2tokens(token_int)
        token = token[:valid_token_num-self.pred_bias]
        # texts = sentence_postprocess(token)
        return token


class ContextualParaformer(Paraformer):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """
    def __init__(self, model_dir: Union[str, Path] = None,
                 batch_size: int = 1,
                 device_id: Union[str, int] = "-1",
                 plot_timestamp_to: str = "",
                 quantize: bool = False,
                 intra_op_num_threads: int = 4,
                 cache_dir: str = None
                 ):

        if not Path(model_dir).exists():
            from modelscope.hub.snapshot_download import snapshot_download
            try:
                model_dir = snapshot_download(model_dir, cache_dir=cache_dir)
            except:
                raise "model_dir must be model_name in modelscope or local path downloaded from modelscope, but is {}".format(model_dir)
        
        model_bb_file = os.path.join(model_dir, 'model_bb.onnx')
        model_eb_file = os.path.join(model_dir, 'model_eb.onnx')

        token_list_file = os.path.join(model_dir, 'tokens.txt')
        self.vocab = {}
        with open(Path(token_list_file), 'r') as fin:
            for i, line in enumerate(fin.readlines()):
                self.vocab[line.strip()] = i

        #if quantize:
        #    model_file = os.path.join(model_dir, 'model_quant.onnx')
        #if not os.path.exists(model_file):
        #    logging.error(".onnx model not exist, please export first.")
            
        config_file = os.path.join(model_dir, 'config.yaml')
        cmvn_file = os.path.join(model_dir, 'am.mvn')
        config = read_yaml(config_file)

        self.converter = TokenIDConverter(config['token_list'])
        self.tokenizer = CharTokenizer()
        self.frontend = WavFrontend(
            cmvn_file=cmvn_file,
            **config['frontend_conf']
        )
        self.ort_infer_bb = OrtInferSession(model_bb_file, device_id, intra_op_num_threads=intra_op_num_threads)
        self.ort_infer_eb = OrtInferSession(model_eb_file, device_id, intra_op_num_threads=intra_op_num_threads)

        self.batch_size = batch_size
        self.plot_timestamp_to = plot_timestamp_to
        if "predictor_bias" in config['model_conf'].keys():
            self.pred_bias = config['model_conf']['predictor_bias']
        else:
            self.pred_bias = 0

    def __call__(self, 
                 wav_content: Union[str, np.ndarray, List[str]], 
                 hotwords: str,
                 **kwargs) -> List:
        # make hotword list
        hotwords, hotwords_length = self.proc_hotword(hotwords)
        [bias_embed] = self.eb_infer(hotwords)
        # index from bias_embed
        bias_embed = bias_embed.transpose(1, 0, 2)
        _ind = np.arange(0, len(hotwords)).tolist()
        bias_embed = bias_embed[_ind, hotwords_length.cpu().numpy().tolist()]
        waveform_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
        waveform_nums = len(waveform_list)
        asr_res = []
        for beg_idx in range(0, waveform_nums, self.batch_size):
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
            bias_embed = np.expand_dims(bias_embed, axis=0)
            bias_embed = np.repeat(bias_embed, feats.shape[0], axis=0)
            try:
                outputs = self.bb_infer(feats, feats_len, bias_embed)
                am_scores, valid_token_lens = outputs[0], outputs[1]
            except ONNXRuntimeError:
                #logging.warning(traceback.format_exc())
                logging.warning("input wav is silence or noise")
                preds = ['']
            else:
                preds = self.decode(am_scores, valid_token_lens)
                for pred in preds:
                    pred = sentence_postprocess(pred)
                    asr_res.append({'preds': pred})
        return asr_res

    def proc_hotword(self, hotwords):
        hotwords = hotwords.split(" ")
        hotwords_length = [len(i) - 1 for i in hotwords]
        hotwords_length.append(0)
        hotwords_length = torch.Tensor(hotwords_length).to(torch.int32)
        # hotwords.append('<s>')
        def word_map(word):
            return torch.tensor([self.vocab[i] for i in word])
        hotword_int = [word_map(i) for i in hotwords]
        hotword_int.append(torch.tensor([1]))
        hotwords = pad_list(hotword_int, pad_value=0, max_len=10)
        return hotwords, hotwords_length

    def bb_infer(self, feats: np.ndarray,
              feats_len: np.ndarray, bias_embed) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_infer_bb([feats, feats_len, bias_embed])
        return outputs

    def eb_infer(self, hotwords):
        outputs = self.ort_infer_eb([hotwords.to(torch.int32).numpy()])
        return outputs

    def decode(self, am_scores: np.ndarray, token_nums: int) -> List[str]:
        return [self.decode_one(am_score, token_num)
                for am_score, token_num in zip(am_scores, token_nums)]

    def decode_one(self,
                   am_score: np.ndarray,
                   valid_token_num: int) -> List[str]:
        yseq = am_score.argmax(axis=-1)
        score = am_score.max(axis=-1)
        score = np.sum(score, axis=-1)

        # pad with mask tokens to ensure compatibility with sos/eos tokens
        # asr_model.sos:1  asr_model.eos:2
        yseq = np.array([1] + yseq.tolist() + [2])
        hyp = Hypothesis(yseq=yseq, score=score)

        # remove sos/eos and get results
        last_pos = -1
        token_int = hyp.yseq[1:last_pos].tolist()

        # remove blank symbol id, which is assumed to be 0
        token_int = list(filter(lambda x: x not in (0, 2), token_int))

        # Change integer-ids to tokens
        token = self.converter.ids2tokens(token_int)
        token = token[:valid_token_num-self.pred_bias]
        # texts = sentence_postprocess(token)
        return token


class ContextualParaformer_v2(Paraformer):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """
    def __init__(self, model_dir: Union[str, Path] = None,
                 batch_size: int = 1,
                 device_id: Union[str, int] = "-1",
                 plot_timestamp_to: str = "",
                 quantize: bool = False,
                 intra_op_num_threads: int = 4,
                 cache_dir: str = None
                 ):

        if not Path(model_dir).exists():
            from modelscope.hub.snapshot_download import snapshot_download
            try:
                model_dir = snapshot_download(model_dir, cache_dir=cache_dir)
            except:
                raise "model_dir must be model_name in modelscope or local path downloaded from modelscope, but is {}".format(model_dir)
        
        model_enc_file = os.path.join(model_dir, 'model_enc.onnx')
        model_dec_file = os.path.join(model_dir, 'model_dec.onnx')
        model_eb_file = os.path.join(model_dir, 'model_eb.onnx')

        token_list_file = os.path.join(model_dir, 'tokens.txt')
        self.vocab = {}
        with open(Path(token_list_file), 'r') as fin:
            for i, line in enumerate(fin.readlines()):
                self.vocab[line.strip()] = i

        #if quantize:
        #    model_file = os.path.join(model_dir, 'model_quant.onnx')
        #if not os.path.exists(model_file):
        #    logging.error(".onnx model not exist, please export first.")
            
        config_file = os.path.join(model_dir, 'config.yaml')
        cmvn_file = os.path.join(model_dir, 'am.mvn')
        config = read_yaml(config_file)

        self.converter = TokenIDConverter(config['token_list'])
        self.tokenizer = CharTokenizer()
        self.frontend = WavFrontend(
            cmvn_file=cmvn_file,
            **config['frontend_conf']
        )
        self.ort_infer_enc = OrtInferSession(model_enc_file, device_id, intra_op_num_threads=intra_op_num_threads)
        self.ort_infer_dec = OrtInferSession(model_dec_file, device_id, intra_op_num_threads=intra_op_num_threads)
        self.ort_infer_eb = OrtInferSession(model_eb_file, device_id, intra_op_num_threads=intra_op_num_threads)

        self.batch_size = batch_size
        self.plot_timestamp_to = plot_timestamp_to
        if "predictor_bias" in config['model_conf'].keys():
            self.pred_bias = config['model_conf']['predictor_bias']
        else:
            self.pred_bias = 0

    def __call__(self, 
                 wav_content: Union[str, np.ndarray, List[str]], 
                 hotwords: str,
                 **kwargs) -> List:
        # make hotword list
        hotwords, hotwords_length = self.proc_hotword(hotwords)
        [bias_embed] = self.eb_infer(hotwords, hotwords_length)
        # index from bias_embed
        bias_embed = bias_embed.transpose(1, 0, 2)
        _ind = np.arange(0, len(hotwords)).tolist()
        bias_embed = bias_embed[_ind, hotwords_length.cpu().numpy().tolist()]
        waveform_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
        waveform_nums = len(waveform_list)
        asr_res = []
        for beg_idx in range(0, waveform_nums, self.batch_size):
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
            bias_embed = np.expand_dims(bias_embed, axis=0)
            bias_embed = np.repeat(bias_embed, feats.shape[0], axis=0)
            # encoder forward
            outputs = self.enc_infer(feats, feats_len)
            enc_output, alphas = outputs
            #outputs2 = self.enc_infer2(feats, feats_len)
            #enc_output2, _ ,alphas2 = outputs2
            #import pdb; pdb.set_trace()
            enc_output_1 = np.copy(enc_output)
            # encoder mask
            mask = ~make_pad_mask(feats_len)
            token_num = alphas.sum(-1)
            enc_output, alphas, token_num = tail_process_fn(enc_output, torch.tensor(alphas), mask=mask)
            acoustic_embeds, fires = cif(enc_output, alphas, threshold=1.0)
            token_num = token_num.floor().type(torch.int32)
            #import pdb; pdb.set_trace()
            # decoder forward
            lmbd = np.array([1.0]*self.batch_size).astype('float32')
            bias_lenth = np.array([len(hotwords)]*self.batch_size).astype("int32")
            outputs = self.dec_infer(enc_output.numpy(), feats_len+1, token_num.int().numpy(), acoustic_embeds.numpy(), bias_embed, lmbd, bias_lenth)
            sampled_ids, am_scores, valid_token_lens = outputs
            # encoder
            '''
            tensor_list = [feats, feats_len, enc_output_1, alphas.numpy()]
            tensor_name_list = ['speech', 'speech_lengths', 'enc_output', 'alphas']
            for tensor, name in zip(tensor_list, tensor_name_list):
                print(tensor.shape, name)
                with open("encoder_output/{}.txt".format(name), 'w') as f:
                    for t in tensor.ravel():
                        f.write("{}\n".format(t))
            # decoder
            tensor_list = [enc_output.numpy(), feats_len+1, token_num.numpy(), acoustic_embeds.numpy(), bias_embed, lmbd, bias_lenth, valid_token_lens, sampled_ids, am_scores]
            tensor_name_list = ['encoder_output', 'encoder_output_lengths', 'token_num.1', 'cif_output', 'bias_embed', 'lmbd', 'bias_length', 'token_num', 'sampled_ids', 'logits']
            for tensor, name in zip(tensor_list, tensor_name_list):
                print(tensor.shape, name)
                with open("decoder_output/{}.txt".format(name), 'w') as f:
                    for t in tensor.ravel():
                        f.write("{}\n".format(t))
            import pdb; pdb.set_trace()
            '''
            # am_scores, valid_token_lens = outputs
            preds = self.decode(am_scores, valid_token_lens)
            for pred in preds:
                pred = sentence_postprocess(pred)
                asr_res.append({'preds': pred})
            '''
            try:
                outputs = self.bb_infer(feats, feats_len, bias_embed)
                am_scores, valid_token_lens = outputs[0], outputs[1]
            except ONNXRuntimeError:
                #logging.warning(traceback.format_exc())
                logging.warning("input wav is silence or noise")
                preds = ['']
            else:
                preds = self.decode(am_scores, valid_token_lens)
                for pred in preds:
                    pred = sentence_postprocess(pred)
                    asr_res.append({'preds': pred})
            '''
        return asr_res

    def proc_hotword(self, hotwords):
        hotwords = hotwords.split(" ")
        hotwords_length = [len(i) - 1 for i in hotwords]
        hotwords_length.append(0)
        hotwords_length = torch.Tensor(hotwords_length).to(torch.int32)
        # hotwords.append('<s>')
        def word_map(word):
            return torch.tensor([self.vocab[i] for i in word])
        hotword_int = [word_map(i) for i in hotwords]
        hotword_int.append(torch.tensor([1]))
        hotwords = pad_list(hotword_int, pad_value=0, max_len=10)
        return hotwords, hotwords_length

    def enc_infer(self, feats: np.ndarray,
              feats_len: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_infer_enc([feats, feats_len])
        return outputs

    def enc_infer2(self, feats: np.ndarray,
              feats_len: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_infer_enc_lhn([feats, feats_len])
        return outputs

    def dec_infer(self, encoder_output, encoder_output_length, token_num, cif_output, lmbd, bias_length, init_cache=None):
        outputs = self.ort_infer_dec([encoder_output, encoder_output_length, token_num, cif_output, lmbd, bias_length, init_cache])
        return outputs

    def eb_infer(self, hotwords, hotwords_length):
        outputs = self.ort_infer_eb([hotwords.to(torch.int32).numpy(), hotwords_length.to(torch.int32).numpy()])
        return outputs

    def decode(self, am_scores: np.ndarray, token_nums: int) -> List[str]:
        return [self.decode_one(am_score, token_num)
                for am_score, token_num in zip(am_scores, token_nums)]

    def decode_one(self,
                   am_score: np.ndarray,
                   valid_token_num: int) -> List[str]:
        yseq = am_score.argmax(axis=-1)
        score = am_score.max(axis=-1)
        score = np.sum(score, axis=-1)

        # pad with mask tokens to ensure compatibility with sos/eos tokens
        # asr_model.sos:1  asr_model.eos:2
        yseq = np.array([1] + yseq.tolist() + [2])
        hyp = Hypothesis(yseq=yseq, score=score)

        # remove sos/eos and get results
        last_pos = -1
        token_int = hyp.yseq[1:last_pos].tolist()

        # remove blank symbol id, which is assumed to be 0
        token_int = list(filter(lambda x: x not in (0, 2), token_int))

        # Change integer-ids to tokens
        token = self.converter.ids2tokens(token_int)
        token = token[:valid_token_num-self.pred_bias]
        # texts = sentence_postprocess(token)
        return token


def cif(hidden, alphas, threshold: float):
    batch_size, len_time, hidden_size = hidden.size()
    threshold = torch.tensor([threshold], dtype=alphas.dtype).to(alphas.device)
    
    # loop varss
    integrate = torch.zeros([batch_size], dtype=alphas.dtype, device=hidden.device)
    frame = torch.zeros([batch_size, hidden_size], dtype=hidden.dtype, device=hidden.device)
    # intermediate vars along time
    list_fires = []
    list_frames = []
    
    for t in range(len_time):
        alpha = alphas[:, t]
        distribution_completion = torch.ones([batch_size], dtype=alphas.dtype, device=hidden.device) - integrate
        integrate += alpha
        list_fires.append(integrate)
        fire_place = integrate >= threshold
        integrate = torch.where(fire_place,
                                integrate - torch.ones([batch_size], dtype=alphas.dtype, device=hidden.device),
                                integrate)
        cur = torch.where(fire_place,
                          distribution_completion,
                          alpha)
        remainds = alpha - cur
        frame += cur[:, None] * hidden[:, t, :]
        list_frames.append(frame)
        frame = torch.where(fire_place[:, None].repeat(1, hidden_size),
                            remainds[:, None] * hidden[:, t, :],
                            frame)
    fires = torch.stack(list_fires, 1)
    frames = torch.stack(list_frames, 1)
    fire_idxs = fires >= threshold
    frame_fires = torch.zeros_like(hidden)
    max_label_len = frames[0, fire_idxs[0]].size(0)
    for b in range(batch_size):
        frame_fire = frames[b, fire_idxs[b]]
        frame_len = frame_fire.size(0)
        frame_fires[b, :frame_len, :] = frame_fire
        if frame_len >= max_label_len:
            max_label_len = frame_len
    frame_fires = frame_fires[:, :max_label_len, :]
    return frame_fires, fires


def tail_process_fn(hidden, alphas, token_num=None, mask=None):
    hidden = torch.tensor(hidden)
    b, t, d = hidden.shape
    tail_threshold = 0.45
    
    zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
    ones_t = torch.ones_like(zeros_t)

    mask_1 = torch.cat([mask, zeros_t], dim=1)
    mask_2 = torch.cat([ones_t, mask], dim=1)
    mask = mask_2 - mask_1
    tail_threshold = mask * tail_threshold
    alphas = torch.cat([alphas, zeros_t], dim=1)
    alphas = torch.add(alphas, tail_threshold)

    zeros = torch.zeros((b, 1, d), dtype=torch.float32).to(alphas.device)
    hidden = torch.cat([hidden, zeros], dim=1)
    token_num = alphas.sum(dim=-1)
    token_num_floor = torch.floor(token_num)
    
    return hidden, alphas, token_num_floor


class ContextualParaformer_v2_chunk(Paraformer):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """
    def __init__(self, model_dir: Union[str, Path] = None,
                 batch_size: int = 1,
                 device_id: Union[str, int] = "-1",
                 plot_timestamp_to: str = "",
                 quantize: bool = False,
                 intra_op_num_threads: int = 4,
                 cache_dir: str = None
                 ):

        if not Path(model_dir).exists():
            from modelscope.hub.snapshot_download import snapshot_download
            try:
                model_dir = snapshot_download(model_dir, cache_dir=cache_dir)
            except:
                raise "model_dir must be model_name in modelscope or local path downloaded from modelscope, but is {}".format(model_dir)
        
        model_enc_file = os.path.join(model_dir, 'model_enc.onnx')
        model_dec_file = os.path.join(model_dir, 'model_dec.onnx')
        model_eb_file = os.path.join(model_dir, 'model_eb.onnx')

        token_list_file = os.path.join(model_dir, 'tokens.txt')
        self.vocab = {}
        with open(Path(token_list_file), 'r') as fin:
            for i, line in enumerate(fin.readlines()):
                self.vocab[line.strip()] = i

        #if quantize:
        #    model_file = os.path.join(model_dir, 'model_quant.onnx')
        #if not os.path.exists(model_file):
        #    logging.error(".onnx model not exist, please export first.")
            
        config_file = os.path.join(model_dir, 'config.yaml')
        cmvn_file = os.path.join(model_dir, 'am.mvn')
        config = read_yaml(config_file)

        self.converter = TokenIDConverter(config['token_list'])
        self.tokenizer = CharTokenizer()
        self.frontend = WavFrontend(
            cmvn_file=cmvn_file,
            **config['frontend_conf']
        )
        self.ort_infer_enc = OrtInferSession(model_enc_file, device_id, intra_op_num_threads=intra_op_num_threads)
        self.ort_infer_dec = OrtInferSession(model_dec_file, device_id, intra_op_num_threads=intra_op_num_threads)
        self.ort_infer_eb = OrtInferSession(model_eb_file, device_id, intra_op_num_threads=intra_op_num_threads)

        self.batch_size = batch_size
        self.plot_timestamp_to = plot_timestamp_to
        if "predictor_bias" in config['model_conf'].keys():
            self.pred_bias = config['model_conf']['predictor_bias']
        else:
            self.pred_bias = 0
        self.pe = SinusoidalPositionEncoderOnline()
        self.encoder_output_size = 512
        self.chunk_size = [15, 120, 15]
        self.feats_dims = 560

    def __call__(self, 
                 wav_content: Union[str, np.ndarray, List[str]], 
                 hotwords: str,
                 **kwargs) -> List:
        data_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
        id_list = [item[0] for item in data_list]
        waveform_list = [item[1] for item in data_list]
        waveform_nums = len(waveform_list)
        asr_res = []
        feats_list, feats_len_list = self.extract_feat(waveform_list)
        # make hotword list
        hotwords, hotwords_length = self.proc_hotword(hotwords)
        [bias_embed] = self.eb_infer(hotwords, hotwords_length)
        # index from bias_embed
        bias_embed = bias_embed.transpose(1, 0, 2)
        _ind = np.arange(0, len(hotwords)).tolist()
        bias_embed = bias_embed[_ind, hotwords_length.cpu().numpy().tolist()]
        bias_embed = np.expand_dims(bias_embed, axis=0)
        asr_res = []
        for beg_idx in range(0, waveform_nums, self.batch_size):
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            max_chunk_num = 0
            cache = []
            id_list_chunk = []
            for i in range(end_idx - beg_idx):
                cache_tmp = self.prepare_cache()
                cache_tmp["chunk_num"] = math.ceil(feats_len_list[beg_idx+i] / self.chunk_size[1])
                if cache_tmp["chunk_num"] > max_chunk_num:
                    max_chunk_num = cache_tmp["chunk_num"]
                cache.append(cache_tmp)
                id_list_chunk.append(id_list[i])
            feats = feats_list[beg_idx:end_idx]
            # forward encoder
            for i in range(max_chunk_num):
                feats_chunk = []
                feats_chunk_lens = []
                for j in range(end_idx - beg_idx):
                    if i+1 > cache[j]["chunk_num"]:
                        cache[j]["last_chunk"] = True
                        continue
                    else:
                        feats_tmp = feats[j][i*self.chunk_size[1]:(i+1)*self.chunk_size[1]]
                        feats_tmp = np.concatenate((cache[j]["feats"], feats_tmp), axis=0)
                        cache[j]["feats"] = feats_tmp[-(self.chunk_size[0] + self.chunk_size[2]):]
                        if i+1 == cache[j]["chunk_num"]:
                            cache[j]["last_chunk"] = True

                        feats_chunk.append(feats_tmp)
                        feats_chunk_lens.append(feats_tmp.shape[0])

                feats_chunk = self.pad_feats(feats_chunk, max(feats_chunk_lens))
                feats_chunk_lens = np.array(feats_chunk_lens).astype(np.int32)
                enc, alphas = self.encode_chunk(feats_chunk, feats_chunk_lens)
                
                index = 0
                for j in range(end_idx - beg_idx):
                    if i+1 <= cache[j]["chunk_num"]:
                        if cache[j]["last_chunk"]:
                            enc_remove_overlap = enc[index, self.chunk_size[0]:feats_chunk_lens[index], :]
                            alpha_remove_overlap = alphas[index, self.chunk_size[0]:feats_chunk_lens[index]]
                        else:
                            enc_remove_overlap = enc[index, self.chunk_size[0]:sum(self.chunk_size[:2]), :]
                            alpha_remove_overlap = alphas[index, self.chunk_size[0]:sum(self.chunk_size[:2])]

                        if cache[j]["encoder_outputs"] is None:
                            cache[j]["encoder_outputs"] = enc_remove_overlap
                            cache[j]["alphas"] = alpha_remove_overlap
                        else:
                            cache[j]["encoder_outputs"] = np.concatenate((cache[j]["encoder_outputs"],
                                                                            enc_remove_overlap), axis=0)
                            cache[j]["alphas"] = np.concatenate((cache[j]["alphas"], alpha_remove_overlap), axis=0)
                        index = index + 1

            enc_list = []
            alpha_list = []
            max_lens = 0
            for j in range(end_idx - beg_idx):
                enc_list.append(cache[j]["encoder_outputs"])
                alpha_list.append(np.expand_dims(cache[j]["alphas"], axis=1))
                if cache[j]["encoder_outputs"].shape[0] > max_lens:
                    max_lens = cache[j]["encoder_outputs"].shape[0]
            enc = self.pad_feats(enc_list, max_lens)
            enc_lens = [item.shape[0] for item in enc_list]
            enc_lens = np.array(enc_lens).astype(np.int32)
            alpha = self.pad_feats(alpha_list, max_lens)
            alpha = np.squeeze(alpha, axis=-1)
            feats_len = enc_lens
            alphas = alpha
            
            bias_embed = np.repeat(bias_embed, enc.shape[0], axis=0)
            enc_output = enc
            
            mask = ~make_pad_mask(feats_len)
            token_num = alphas.sum(-1)
            enc_output, alphas, token_num = tail_process_fn(enc_output, torch.tensor(alphas), mask=mask)
            acoustic_embeds, fires = cif(enc_output, alphas, threshold=1.0)
            token_num = token_num.floor().type(torch.int32)
            #import pdb; pdb.set_trace()
            # decoder forward
            lmbd = np.array([1.0]*self.batch_size).astype('float32')
            bias_lenth = np.array([len(hotwords)]*self.batch_size).astype("int32")
            outputs = self.dec_infer(enc_output.numpy(), feats_len+1, token_num.int().numpy(), acoustic_embeds.numpy(), bias_embed, lmbd, bias_lenth)
            sampled_ids, am_scores, valid_token_lens = outputs
            
            # am_scores, valid_token_lens = outputs
            preds = self.decode(am_scores, valid_token_lens)
            for pred in preds:
                pred = sentence_postprocess(pred)
                asr_res.append({'preds': pred})
            '''
            try:
                outputs = self.bb_infer(feats, feats_len, bias_embed)
                am_scores, valid_token_lens = outputs[0], outputs[1]
            except ONNXRuntimeError:
                #logging.warning(traceback.format_exc())
                logging.warning("input wav is silence or noise")
                preds = ['']
            else:
                preds = self.decode(am_scores, valid_token_lens)
                for pred in preds:
                    pred = sentence_postprocess(pred)
                    asr_res.append({'preds': pred})
            '''
        return asr_res

    def proc_hotword(self, hotwords):
        hotwords = hotwords.split(" ")
        hotwords_length = [len(i) - 1 for i in hotwords]
        hotwords_length.append(0)
        hotwords_length = torch.Tensor(hotwords_length).to(torch.int32)
        # hotwords.append('<s>')
        def word_map(word):
            return torch.tensor([self.vocab[i] for i in word])
        hotword_int = [word_map(i) for i in hotwords]
        hotword_int.append(torch.tensor([1]))
        hotwords = pad_list(hotword_int, pad_value=0, max_len=10)
        return hotwords, hotwords_length

    def enc_infer(self, feats: np.ndarray,
              feats_len: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_infer_enc([feats, feats_len])
        return outputs

    def enc_infer2(self, feats: np.ndarray,
              feats_len: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_infer_enc_lhn([feats, feats_len])
        return outputs

    def dec_infer(self, encoder_output, encoder_output_length, token_num, cif_output, lmbd, bias_length, init_cache=None):
        outputs = self.ort_infer_dec([encoder_output, encoder_output_length, token_num, cif_output, lmbd, bias_length, init_cache])
        return outputs

    def eb_infer(self, hotwords, hotwords_length):
        outputs = self.ort_infer_eb([hotwords.to(torch.int32).numpy(), hotwords_length.to(torch.int32).numpy()])
        return outputs

    def decode(self, am_scores: np.ndarray, token_nums: int) -> List[str]:
        return [self.decode_one(am_score, token_num)
                for am_score, token_num in zip(am_scores, token_nums)]

    def decode_one(self,
                   am_score: np.ndarray,
                   valid_token_num: int) -> List[str]:
        yseq = am_score.argmax(axis=-1)
        score = am_score.max(axis=-1)
        score = np.sum(score, axis=-1)

        # pad with mask tokens to ensure compatibility with sos/eos tokens
        # asr_model.sos:1  asr_model.eos:2
        yseq = np.array([1] + yseq.tolist() + [2])
        hyp = Hypothesis(yseq=yseq, score=score)

        # remove sos/eos and get results
        last_pos = -1
        token_int = hyp.yseq[1:last_pos].tolist()

        # remove blank symbol id, which is assumed to be 0
        token_int = list(filter(lambda x: x not in (0, 2), token_int))

        # Change integer-ids to tokens
        token = self.converter.ids2tokens(token_int)
        token = token[:valid_token_num-self.pred_bias]
        # texts = sentence_postprocess(token)
        return token
    def pad_feats_(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, 'constant', constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats
    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, 'constant', constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def encode_chunk(self, feats: np.ndarray, feats_len: np.ndarray):
        # encoder forward
        #enc_input = [feats, feats_len]
        #enc, enc_lens, cif_alphas = self.ort_encoder_infer(enc_input)
        enc, cif_alphas = self.ort_infer_enc([feats, feats_len])
        return enc, cif_alphas
    
    def extract_feat(self,
                     waveform_list: List[np.ndarray]
                     ) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_len = [], []
        for waveform in waveform_list:
            speech, _ = self.frontend.fbank(waveform)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feat *= self.encoder_output_size ** 0.5
            feat = self.pe.forward(np.expand_dims(feat, axis=0))
            feats.append(np.squeeze(feat, axis=0))
            feats_len.append(feat_len)
        return feats, feats_len
    
    def load_data(self,
                  wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List:
        def load_wav(path: str) -> np.ndarray:
            waveform, _ = librosa.load(path, sr=fs)
            return waveform

        if isinstance(wav_content, np.ndarray):
            return [("test", wav_content)]

        if isinstance(wav_content, str):
            if not os.path.exists(wav_content):
                raise TypeError("The file of {} is not exits".format(str))
            if wav_content.endswith(".wav"):
                return [("test", load_wav(wav_content))]
            elif wav_content.endswith(".scp"):
                with open(wav_content, "r") as file:
                    lines = file.readlines()
                return [(item.strip().split()[0], load_wav(item.strip().split()[1])) for item in lines]

        if isinstance(wav_content, list):
            return [("test", load_wav(path)) for path in wav_content]
        raise TypeError(
            f'The type of {wav_content} is not in [str, np.ndarray]')
    
    def prepare_cache(self):
        cache = {}
        cache["encoder_outputs"] = None
        cache["alphas"] = None
        cache["last_chunk"] = False
        cache["feats"] = np.zeros((self.chunk_size[0] + self.chunk_size[2], self.feats_dims)).astype(np.float32)
        return cache


def cif(hidden, alphas, threshold: float):
    batch_size, len_time, hidden_size = hidden.size()
    threshold = torch.tensor([threshold], dtype=alphas.dtype).to(alphas.device)
    
    # loop varss
    integrate = torch.zeros([batch_size], dtype=alphas.dtype, device=hidden.device)
    frame = torch.zeros([batch_size, hidden_size], dtype=hidden.dtype, device=hidden.device)
    # intermediate vars along time
    list_fires = []
    list_frames = []
    
    for t in range(len_time):
        alpha = alphas[:, t]
        distribution_completion = torch.ones([batch_size], dtype=alphas.dtype, device=hidden.device) - integrate
        integrate += alpha
        list_fires.append(integrate)
        fire_place = integrate >= threshold
        integrate = torch.where(fire_place,
                                integrate - torch.ones([batch_size], dtype=alphas.dtype, device=hidden.device),
                                integrate)
        cur = torch.where(fire_place,
                          distribution_completion,
                          alpha)
        remainds = alpha - cur
        frame += cur[:, None] * hidden[:, t, :]
        list_frames.append(frame)
        frame = torch.where(fire_place[:, None].repeat(1, hidden_size),
                            remainds[:, None] * hidden[:, t, :],
                            frame)
    fires = torch.stack(list_fires, 1)
    frames = torch.stack(list_frames, 1)
    fire_idxs = fires >= threshold
    frame_fires = torch.zeros_like(hidden)
    max_label_len = frames[0, fire_idxs[0]].size(0)
    for b in range(batch_size):
        frame_fire = frames[b, fire_idxs[b]]
        frame_len = frame_fire.size(0)
        frame_fires[b, :frame_len, :] = frame_fire
        if frame_len >= max_label_len:
            max_label_len = frame_len
    frame_fires = frame_fires[:, :max_label_len, :]
    return frame_fires, fires


def tail_process_fn(hidden, alphas, token_num=None, mask=None):
    hidden = torch.tensor(hidden)
    b, t, d = hidden.shape
    tail_threshold = 0.45
    
    zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
    ones_t = torch.ones_like(zeros_t)

    mask_1 = torch.cat([mask, zeros_t], dim=1)
    mask_2 = torch.cat([ones_t, mask], dim=1)
    mask = mask_2 - mask_1
    tail_threshold = mask * tail_threshold
    alphas = torch.cat([alphas, zeros_t], dim=1)
    alphas = torch.add(alphas, tail_threshold)

    zeros = torch.zeros((b, 1, d), dtype=torch.float32).to(alphas.device)
    hidden = torch.cat([hidden, zeros], dim=1)
    token_num = alphas.sum(dim=-1)
    token_num_floor = torch.floor(token_num)
    
    return hidden, alphas, token_num_floor