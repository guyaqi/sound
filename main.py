import librosa
import librosa.display
from matplotlib import pyplot as plt
import soundfile

import logging
logging.basicConfig(level=logging.NOTSET)


y, sr = librosa.load('./data/long_001.mp3', sr=None)
logging.info(f'采样率: {sr}')

# show waveform
def gen_waveform():
  plt.figure()

  # use waveplot to gen waveform
  librosa.display.waveplot(y, sr)

  plt.title('waveform')
  plt.savefig("./out/waveform.png")

def gen_specs():
  melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
  logmelspec = librosa.power_to_db(melspec)
  plt.figure()
  librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
  plt.title('spec')
  plt.savefig("./out/spec.png")

def gen_sound():
  para = 3
  weight = 1/para

  total_len = len(y)
  priece = int(total_len/para)
  new_y = y[0: priece] * 0
  for i in range(para):
    new_y += y[priece * i: priece * (i+1)] * weight
  
  soundfile.write('./out/1.wav', new_y, sr)

# gen_waveform()
# gen_specs()
gen_sound()