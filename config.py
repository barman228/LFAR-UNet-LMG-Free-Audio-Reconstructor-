sample_rate_low = 16000      # низкое качество
sample_rate_high = 44100     # высокое качество

n_fft = 1024
hop_length = 256
win_length = 1024

chunk_seconds = 2.0          # длина одного обучающего сэмпла
batch_size = 1
epochs = 20                  # МОЖЕШЬ МЕНЯТЬ
num_workers = 2

model_channels = 16          # баланс скорости/качества
device = "cpu"