import torchaudio
import sounddevice as sd
from speechbrain.inference.separation import SepformerSeparation as separator
from scipy.io.wavfile import write


def record(duration=5, sr=16000, filename='recorded_input.wav'):
    print("Recording...")
    recorded_audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished")
    write(filename, sr, recorded_audio)


def preprocess_audio(wf, sr):
    resample_transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=8000)
    new_waveform = resample_transform(wf)
    new_waveform = new_waveform / new_waveform.abs().max()

    return new_waveform


def play_audio(filename="enhanced.wav"):
    print("Playing enhanced audio...")
    enhanced_audio, _ = torchaudio.load(filename)
    if enhanced_audio.shape[0] > 2:
        enhanced_audio = enhanced_audio[:2, :]
    elif enhanced_audio.shape[0] == 1:
        enhanced_audio = enhanced_audio.repeat(2, 1)

    sd.play(enhanced_audio.numpy().T, 8000)
    sd.wait()


input_filename = 'recorded_input.wav'
output_filename = 'enhanced.wav'

record(filename=input_filename)
waveform, sample_rate = torchaudio.load(input_filename)
waveform = preprocess_audio(waveform, sample_rate)

# for using GPU, pass run_opts={"device":"cuda"} to .from_hparams
model = separator.from_hparams(source="speechbrain/sepformer-wham-enhancement",
                               savedir='pretrained_models/sepformer-wham-enhancement')

est_sources = model.separate_batch(waveform)

torchaudio.save(output_filename, est_sources[:, :, 0].detach(), 8000)
play_audio()