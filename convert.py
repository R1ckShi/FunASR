"""Tensorflow to kaldi conveter.
   convert clas encoder (only embedding and lstm) to kaldi format
   by jiangyu.xzy
"""
from __future__ import print_function, division
import os
import struct
import numpy as np
import argparse
from tensorflow.python import pywrap_tensorflow
np.set_printoptions(threshold=np.inf)

# read torch model
import torch
torch_model = torch.load("./model.pb")
torch_embedding_matrix = torch_model['decoder.embed.0.weight'].numpy() # 8404*512
torch_lstm_weight1 = torch_model['bias_encoder.weight_ih_l0'].numpy().transpose() # 2048*512.T
torch_lstm_weight2 = torch_model['bias_encoder.weight_hh_l0'].numpy().transpose() # 2048*512.T
torch_lstm_bias1 = torch_model['bias_encoder.bias_ih_l0'].numpy() # 2048
torch_lstm_bias2 = torch_model['bias_encoder.bias_ih_l0'].numpy() # 2048
# reorder
for m in [torch_lstm_weight1, torch_lstm_weight2]:
    din, dout = m.shape
    m = m.reshape([din, 4, dout // 4])
    ori = np.copy(m)
    m[:,1] = ori[:,0]
    m[:,0] = ori[:,2]
    m[:,2] = ori[:,1]
    m = m.reshape([din, dout])
for m in [torch_lstm_bias1, torch_lstm_bias2]:
    dout = m.shape
    m = m.reshape([4, dout // 4])
    ori = np.copy(m)
    m[1] = ori[0]
    m[0] = ori[2]
    m[2] = ori[1]
    m = m.reshape([dout])
torch_lstm_bias = torch_lstm_bias1 + torch_lstm_bias2


def write_kaldi_matrix(f, name_len, tensor_name, row_num, col_num, data_type, tensor_var, is_bin=False):
  if is_bin:
    # print("write bin")
    name_len = struct.pack('i', name_len)
    tensor_name = tensor_name
    row_num = struct.pack('i', row_num)
    col_num = struct.pack('i', col_num)
    data_type = struct.pack('i', data_type)
    tensor_var = tensor_var.tobytes()
  else:
    # print("write txt")
    name_len = "{}".format(name_len)
    tensor_name = "{}".format(tensor_name)
    row_num = "{}".format(row_num)
    col_num = "{}".format(col_num)
    data_type = "{}".format(data_type)
    tensor_var = "{}".format(tensor_var)
    # tensor_var = tensor_var.tostring()
  print(tensor_name)
  f.write(name_len)
  f.write(tensor_name)
  f.write(row_num)
  f.write(col_num)
  f.write(data_type)
  f.write(tensor_var)
  f.flush()

def _kernel(f, var_dict, layer_name, layer_key, sub_key="kernel", tensor_name_prefix="decoder_ffn", is_bin=False):
  tensor_name = '{}_{}_{}_'.format(tensor_name_prefix, layer_key, sub_key)
  tensor_key = "{}/{}/{}".format(layer_name, layer_key, sub_key)
  if tensor_key not in var_dict:
    print("{} is not exist!".format(tensor_key))
  else:
    tensor_var = var_dict[tensor_key]
    print("embedding: tf shape {} -> torch shape".format(tensor_var.shape, torch_embedding_matrix.shape))
    tensor_var = torch_embedding_matrix
    shape = tensor_var.shape
    map_dict = "{} -> {} {}".format(tensor_key, tensor_name, shape)
    print(map_dict)
    if len(shape) > 2:
        row_num = shape[1] * shape[0]
        col_num = shape[2]
    else:
      row_num = shape[0]
      col_num = shape[1]
    name_len = len(tensor_name)
    data_type = 32
    write_kaldi_matrix(f, name_len, tensor_name, row_num, col_num, data_type, tensor_var.ravel(), is_bin=is_bin)

def write_clas_embeds(f, var_dict, layer_name, has_bias=False, is_bin=False):
  tensor_name_prefix = "contextual_encoder"

  layer_name_w = "{}/contextual_encoder".format(layer_name)
  layer_key = "inputter_0"
  sub_key = "w_char_embs"

  if "{}/{}/{}".format(layer_name_w, layer_key, sub_key) not in var_dict.keys():
    tmp_tensor_name = "{}/{}".format(layer_name_w, sub_key)
    assert tmp_tensor_name in var_dict.keys()
    var_dict["{}/{}/{}".format(layer_name_w, layer_key, sub_key)] = var_dict[tmp_tensor_name]

  _kernel(f, var_dict, layer_name_w, layer_key, sub_key=sub_key, tensor_name_prefix=tensor_name_prefix, is_bin=is_bin)


def load_ckpt(checkpoint_path):
  reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
  var_to_shape_map = reader.get_variable_to_shape_map()

  var_dict = dict()
  for var_name in sorted(var_to_shape_map):
    if "Adam" in var_name:
      continue
    tensor = reader.get_tensor(var_name)
    #print("in ckpt: {}, {}".format(var_name, tensor.shape))
    # print(tensor)
    var_dict[var_name] = tensor

  return var_dict

def write_context(f, context, is_bin=False):
  if is_bin:
    context = struct.pack('i', context)
  else:
    context = "{}".format(context)
  f.write(context)
  f.flush()

def write_layer_head(f, layer_name_, is_bin=False):
  context = "<Layer>"
  write_context(f, context, is_bin=False)
  context = len(layer_name_)
  write_context(f, context, is_bin=is_bin)
  write_context(f, layer_name_, is_bin=False)

def write_layer_end(f, context="</Layer>", is_bin=False):
  write_context(f, context, is_bin=False)
  #print(context)

def write_net_head(f,context="<Net>", is_bin=False):
  write_context(f, context, is_bin=False)
  #print(context)

def write_net_end(f, context="</Net>", is_bin=False):
  write_context(f, context, is_bin=False)
  #print(context)

def _write_lstm_kernel(f, var_dict, layer_name, sub_layer_name, is_bin=False):
  tensor_key = "{}/{}".format(layer_name, sub_layer_name)
  if tensor_key not in var_dict:
    print("{} is not exist!".format(tensor_key))
  else:
    print("{}".format(tensor_key))
    tensor_var = var_dict[tensor_key] # 640*1280
    import pdb; pdb.set_trace()
    row, col = tensor_var.shape
    d_lstm = col // 4
    d_in = row - d_lstm
    tensor_var = tensor_var.reshape([row, 4, d_lstm])
    tensor_var_ori = np.copy(tensor_var)
    tensor_var[:, 0, :] = np.copy(tensor_var_ori[:, 1, :])
    tensor_var[:, 1, :] = np.copy(tensor_var_ori[:, 0, :])
    tensor_var = tensor_var.reshape([row, col])
    wf = tensor_var[0: d_in, :] # 320*1280
    rw = tensor_var[d_in:, :] # 320*1280
    print("lstm weight1: tf shape {} -> torch shape {}".format(wf.shape, torch_lstm_weight1.shape))
    print("lstm weight2: tf shape {} -> torch shape {}".format(rw.shape, torch_lstm_weight2.shape))
    wf = torch_lstm_weight1 # 512*2048
    rw = torch_lstm_weight2 # 512*2048
    _write_matrix(f, wf, is_bin=is_bin, t=True)
    _write_matrix(f, rw, is_bin=is_bin, t=True)

def _write_lstm_bias(f, var_dict, layer_name, sub_layer_name, is_bin=False):
  tensor_key = "{}/{}".format(layer_name, sub_layer_name)
  if tensor_key not in var_dict:
    print("{} is not exist!".format(tensor_key))
  else:
    print("{}".format(tensor_key))
    tensor_var = var_dict[tensor_key]
    col = tensor_var.shape[0]
    d_lstm = col // 4
    tensor_var = tensor_var.reshape([4, d_lstm])
    tensor_var[2] += np.ones_like(tensor_var[2])
    tensor_var_ori = np.copy(tensor_var)
    tensor_var[0, :] = np.copy(tensor_var_ori[1, :])
    tensor_var[1, :] = np.copy(tensor_var_ori[0, :])
    bf = tensor_var.reshape([col]) # 1280
    # torch lstm bias
    bs = torch_lstm_bias
    _write_matrix(f, bf, is_bin=is_bin, t=True)

def _write_matrix(f, tensor_var, is_bin=False, t=True):
  shape = tensor_var.shape
  if len(shape) == 1:
    row_num = shape[0]
    col_num = 1
  elif len(shape) == 2:
    row_num = shape[0]
    col_num = shape[1]
  else:
    write_context(f, shape[0], is_bin=is_bin)
    row_num = shape[1] * shape[0]
    col_num = shape[2]
  tensor_var = tensor_var.reshape([row_num, col_num])
  if t:
    tensor_var = tensor_var.T
  col_num, row_num = tensor_var.shape
  write_kaldi_matrix_no_name(f, row_num, col_num, tensor_var.ravel(), is_bin=is_bin)

def write_kaldi_matrix_no_name(f, row_num, col_num, tensor_var, is_bin=False):
  print("({}, {})".format(row_num, col_num))
  if is_bin:
    row_num = struct.pack('i', row_num)
    col_num = struct.pack('i', col_num)
    tensor_var = tensor_var.tobytes()
  else:
    row_num = "{}".format(row_num)
    col_num = "{}".format(col_num)
    tensor_var = "{}".format(tensor_var)

  f.write(row_num)
  f.write(col_num)
  f.write(tensor_var)
  f.flush()

def _write_clas_lstm(f, var_dict, layer_name, sub_layer_name, is_bin=False):
  layer_name = "{}/{}".format(layer_name, sub_layer_name)

  _write_lstm_kernel(f, var_dict, layer_name, "kernel", is_bin=is_bin)
  _write_lstm_bias(f, var_dict, layer_name, "bias", is_bin=is_bin)


def convert(f, var_dict, is_bin):
  write_net_head(f)

  layer_name_ = "clasembeddingLayer"
  write_layer_head(f, layer_name_, is_bin=is_bin)
  layer_name = "seq2seq"
  write_clas_embeds(f, var_dict, layer_name, is_bin=is_bin)
  write_layer_end(f)

  layer_name_ = "clasLstmLayer"
  write_layer_head(f, layer_name_, is_bin=is_bin)
  layer_name = "seq2seq"
  sub_layer_name = "clas_charrnn"
  layer_name = "{}/{}".format(layer_name, sub_layer_name)
  _write_clas_lstm(f, var_dict, layer_name, "rnn/lstm_cell", is_bin=is_bin)
  write_layer_end(f)

  write_net_end(f)


if __name__ == '__main__':

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--checkpoint_path",
                      default="/home/jiangyu.xzy/exp/aoa_clas/aoa_clas_exp1_1_1/model.ckpt-180000",
                      help=("checkpoint to restore, split by ,"))
  parser.add_argument("--is_bin", type=bool, default=True, help="")
  parser.add_argument("--out_path", default="../tf_to_kaldi_scama_exp26_v2", help=(""))
  parser.add_argument('--file_name', default="clas_encoder.xnn")
  args = parser.parse_args()

  out_path = args.out_path
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  if args.is_bin:
    f = open(os.path.join(out_path, args.file_name), mode='wb')
  else:
    f = open(os.path.join(out_path, args.file_name+'.txt'), mode='w')

  var_dict = load_ckpt(args.checkpoint_path)
  convert(f, var_dict, is_bin=args.is_bin)
  f.close()
