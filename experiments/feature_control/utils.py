import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchaudio

def examine_model(model, batch, sampling_rate, path):
  """
  Evaluate specific samples (batch only, no fallbacks) and export:
    - <idx>_real.wav
    - <idx>_gen.wav
    - <idx>_overview.png   (Top: gen waveform+features+latents; Mid: gen spec; Bot: real spec)
    - <idx>_real_spec.png  (standalone real spectrogram)

  Args:
    model:          your trained model (with .encoder/.decoder/_synthesize/_smooth_latents)
    batch:          tuple (x_audio, x_features) with shapes [B, 1, T] and [B, T_feat, n_features]
    sampling_rate:  int
    path:           output directory (created if missing)
  """
  os.makedirs(path, exist_ok=True)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device).eval()

  # Log-magnitude spectrogram (clearer than power)
  n_fft = 1024
  hop   = 256
  spec_mag = torchaudio.transforms.Spectrogram(
      n_fft=n_fft, hop_length=hop, power=1.0  # magnitude
  ).to(device)
  to_db = torchaudio.transforms.AmplitudeToDB(stype="magnitude").to(device)

  # Unpack batch and move to device
  x_audio, x_features = batch
  x_audio    = x_audio.to(device)      # [B, 1, T]
  x_features = x_features.to(device)   # [B, T_feat, n_features]

  with torch.no_grad():
    # Encode -> sample -> smooth
    mu, scale = model.encoder(x_audio)
    z, _ = model.encoder.reparametrize(mu, scale)
    z = model._smooth_latents(z)

    # Match feature rate expected by decoder
    x_features_res = torch.nn.functional.interpolate(
      x_features.permute(0, 2, 1),
      scale_factor=1 / model.resampling_factor,
      mode='linear',
      align_corners=False
    ).permute(0, 2, 1)

    # Decode -> synthesize (no waveshaping)
    synth_params = model.decoder(x_features_res, z).squeeze(1)
    y_audio = model._synthesize(synth_params)  # [B, 1, T_gen]

    print("x_audio shape:", tuple(x_audio.shape), "y_audio shape:", tuple(y_audio.shape))

    B = x_audio.shape[0]
    feature_names = ['loudness', 'centroid']  # adjust to your feature set

    for j in range(B):
      print(f"Sample {j + 1}:")
      x_audio_j = x_audio[j].detach().cpu().squeeze(0)   # [T]
      y_audio_j = y_audio[j].detach().cpu().squeeze(0)   # [T_gen]

      # Save audio
      real_wav_path = os.path.join(path, f"{j:03d}_real.wav")
      gen_wav_path  = os.path.join(path, f"{j:03d}_gen.wav")
      torchaudio.save(real_wav_path, x_audio_j.unsqueeze(0).to(torch.float32), sampling_rate)
      torchaudio.save(gen_wav_path,  y_audio_j.unsqueeze(0).to(torch.float32), sampling_rate)

      # Arrays for plotting
      gen_waveform = y_audio_j.numpy()
      feat = x_features_res[j].detach().cpu().numpy()    # [T_feat, n_features]
      lat  = z[j].detach().cpu().numpy()                 # [T_lat, n_lat] or [n_lat, T_lat]

      # Time axis for generated waveform
      T_gen = gen_waveform.shape[-1]
      t_gen = np.arange(T_gen) / sampling_rate

      # Helper
      def safe_norm(x):
        x = np.asarray(x)
        x_min, x_max = x.min(), x.max()
        return (x - x_min) / (x_max - x_min + 1e-12)

      # Spectrograms (log-mag dB), auto-scaled for display
      g = y_audio[j:j+1].to(device)  # [1, 1, T_gen]
      r = x_audio[j:j+1].to(device)  # [1, 1, T]

      g_db = to_db(spec_mag(g)).squeeze(0).squeeze(0).detach().cpu().numpy()  # [F, frames]
      r_db = to_db(spec_mag(r)).squeeze(0).squeeze(0).detach().cpu().numpy()

      t_end_g = (g_db.shape[1] - 1) * (hop / sampling_rate) if g_db.shape[1] > 1 else t_gen[-1]
      t_end_r = (r_db.shape[1] - 1) * (hop / sampling_rate) if r_db.shape[1] > 1 else (x_audio_j.shape[-1] / sampling_rate)

      # ---------- 3-panel overview ----------
      fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 2, 2]})
      ax_top, ax_mid, ax_bot = axes
      fig.suptitle(f"Sample {j+1}: Gen Waveform/Features/Latents + Spectrograms (log-mag dB)", y=0.98)

      # Top: ONLY generated waveform + selected features + latents
      ax_top.plot(t_gen, gen_waveform, label='waveform (gen)', alpha=0.85)

      for i, name in enumerate(feature_names):
        if i < feat.shape[1]:
          f = feat[:, i]
          tf = np.linspace(0, t_gen[-1], len(f))
          ax_top.plot(tf, safe_norm(f), label=f'feat:{name} (norm)', linewidth=1.2)

      if lat.ndim == 2:
        T_lat, n_lat = lat.shape
        if T_lat < n_lat:  # likely [n_lat, T_lat]
          lat = lat.T
          T_lat, n_lat = lat.shape
        tl = np.linspace(0, t_gen[-1], T_lat)
        for k in range(min(n_lat, 6)):  # cap to 6 for readability
          ax_top.plot(tl, safe_norm(lat[:, k]), label=f'latent l{k+1} (norm)', linewidth=1)

      ax_top.set_xlim(0, t_gen[-1])
      ax_top.set_xlabel('Time (s)')
      ax_top.set_ylabel('Amplitude / norm')
      ax_top.legend(loc='upper right', ncol=2, fontsize=8)
      ax_top.grid(True, alpha=0.2)

      # Middle: GENERATED spectrogram (log-mag dB)
      im_g = ax_mid.imshow(
        g_db, aspect='auto', origin='lower',
        extent=[0, t_end_g, 0, sampling_rate/2]
      )
      ax_mid.set_xlabel('Time (s)')
      ax_mid.set_ylabel('Frequency (Hz)')
      ax_mid.set_title('Generated Spectrogram (log-mag dB)')
      fig.colorbar(im_g, ax=ax_mid, fraction=0.046, pad=0.02)

      # Bottom: REAL spectrogram (log-mag dB)
      im_r = ax_bot.imshow(
        r_db, aspect='auto', origin='lower',
        extent=[0, t_end_r, 0, sampling_rate/2]
      )
      ax_bot.set_xlabel('Time (s)')
      ax_bot.set_ylabel('Frequency (Hz)')
      ax_bot.set_title('Real Spectrogram (log-mag dB)')
      fig.colorbar(im_r, ax=ax_bot, fraction=0.046, pad=0.02)

      plt.tight_layout(rect=[0, 0, 1, 0.96])

      overview_path  = os.path.join(path, f"{j:03d}_overview.png")
      real_spec_path = os.path.join(path, f"{j:03d}_real_spec.png")
      fig.savefig(overview_path, dpi=150)
      plt.close(fig)

      # Standalone real spectrogram file (optional but handy)
      fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
      im2 = ax2.imshow(
        r_db, aspect='auto', origin='lower',
        extent=[0, t_end_r, 0, sampling_rate/2]
      )
      ax2.set_xlabel('Time (s)')
      ax2.set_ylabel('Frequency (Hz)')
      ax2.set_title('Real Spectrogram (log-mag dB)')
      fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
      plt.tight_layout()
      fig2.savefig(real_spec_path, dpi=150)
      plt.close(fig2)

      print(f"Saved: {real_wav_path}, {gen_wav_path}, {overview_path}, {real_spec_path}")
