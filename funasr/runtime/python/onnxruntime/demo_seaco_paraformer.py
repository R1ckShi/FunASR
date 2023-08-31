# coding:utf-8
from funasr_onnx import SeACoParaformer2 as SeACoParaformer
from pathlib import Path

model_dir = "/Users/shixian/code/speech_seaco_paraformer3"
model = SeACoParaformer(model_dir, batch_size=1)
# model = Paraformer(model_dir, batch_size=1, device_id=0)  # gpu

# when using paraformer-large-vad-punc model, you can set plot_timestamp_to="./xx.png" to get figure of alignment besides timestamps
# model = Paraformer(model_dir, batch_size=1, plot_timestamp_to="test.png")

# wav_path = ['{}/.cache/modelscope/hub/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/example/asr_example.wav'.format(Path.home())]
wav_path = ['/Users/shixian/Downloads/sac_test.wav']
hotwords = ''
# hotwords = '戒色'
result = model(wav_path, hotwords)
print(result)
