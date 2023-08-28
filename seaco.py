from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

param_dict = dict()
param_dict['hotword'] = "邓玉嵩"

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model="/Users/shixian/code/speech_seaco_paraformer",
    param_dict=param_dict)

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_hotword.wav')
print(rec_result)
