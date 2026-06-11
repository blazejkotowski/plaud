from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any, Optional, Tuple, Dict

import lmdb
import torch

from ddsp.audio_feature_dataset import AudioFeatureDataset
from ddsp.ddsp import DDSP
from ddsp.interfaces import ControlSpace
from ddsp.latent_compressor import LatentCompressor
from ddsp.utils import find_checkpoint


def _control_space_signature(control_space: ControlSpace) -> str:
    return ";".join(
        [
            f"{f.name}:{f.source}:{int(f.dim)}:{getattr(f, 'extractor', None)}:{sorted((f.params or {}).items())}"
            for f in control_space.fields
        ]
    )


def _create_cache_key(
    dataset_path: str,
    n_signal: int,
    sampling_rate: int,
    resampling_factor: int,
    n_channels: int,
    control_space: ControlSpace,
    ddsp_checkpoint_path: str,
    seq_len: int,
    stride_factor: float,
) -> str:
    """Create a stable cache key similar to AudioFeatureDataset.

    Uses cheap checkpoint identity (path + mtime + size) to avoid hashing large files.
    """
    ckpt_stat = os.stat(ddsp_checkpoint_path)
    ckpt_sig = f"{ddsp_checkpoint_path}:{int(ckpt_stat.st_mtime)}:{int(ckpt_stat.st_size)}"
    field_sig = _control_space_signature(control_space)
    key_string = f"{dataset_path}_{n_signal}_{sampling_rate}_{resampling_factor}_{n_channels}_{seq_len}_{stride_factor}_{field_sig}_{ckpt_sig}"
    return hashlib.md5(key_string.encode()).hexdigest()[:8]


def _default_cache_path(dataset_path: str, cache_key: str) -> str:
    """Follow AudioFeatureDataset convention: cache lives next to dataset directory."""
    base_dir = os.path.dirname(dataset_path)
    return os.path.join(base_dir, f"prior_cache_{cache_key}.lmdb")


def infer_ddsp_checkpoint_path(cfg: Any) -> str:
    """Infer DDSP checkpoint from experiment name/training_dir (no extra Hydra knobs)."""
    model_name = cfg.experiment.name
    training_dir = cfg.experiment.training_dir
    synth_training_path = os.path.join(training_dir, "synth", model_name)
    ckpt = find_checkpoint(synth_training_path, return_none=True, typ="best")
    if ckpt is None:
        ckpt = find_checkpoint(synth_training_path, return_none=False, typ="last")
    return ckpt


def ensure_prior_controls_lmdb(
    *,
    audio_ds: AudioFeatureDataset,
    ddsp: DDSP,
    out_path: str,
    seq_len: int,
    stride_factor: float,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Build LMDB cache at out_path if it doesn't exist; otherwise return metadata.

    LMDB schema:
      - key 'metadata' -> pickled dict
      - keys 'controls:{idx:08d}' -> float32 bytes of shape [seq_len, D]
    """
    if os.path.exists(out_path):
        env = lmdb.open(out_path, readonly=True, lock=False, readahead=True, subdir=True)
        with env.begin() as txn:
            meta_buf = txn.get(b"metadata")
        env.close()
        if meta_buf is None:
            raise RuntimeError(f"Existing prior cache missing metadata: {out_path}")
        return {"path": out_path, **pickle.loads(meta_buf), "rebuilt": False}

    print("Building prior controls LMDB cache at:", out_path)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ddsp = ddsp.to(dev)
    # ddsp is a LightningModule in real use, but tests may pass a minimal dummy.
    if hasattr(ddsp, "eval"):
        ddsp.eval()

    encoder = getattr(ddsp, "encoder", None)
    if encoder is not None and hasattr(encoder, "streaming"):
        encoder.streaming = False

    audio = audio_ds._audio.to(dev)
    features = audio_ds._features.to(dev)  # audio-rate features
    sr = getattr(audio_ds, "_sampling_rate", None)
    if sr is None:
        sr = getattr(ddsp, "fs", 44100)
    sr = int(sr)
    rf = int(ddsp.resampling_factor)

    stride = max(1, int(seq_len * float(stride_factor)))

    # LMDB writer
    env = lmdb.open(
        out_path,
        map_size=16 * 1024**3,  # 16 GiB to start; auto-grow on MapFullError
        readahead=False,
        writemap=False,
        max_dbs=1,
        # This is a derived cache; prioritize throughput over durability.
        # A crash during build just means rebuilding.
        sync=False,
        metasync=False,
        map_async=True,
        subdir=True,
    )

    idx = 0
    num_controls: Optional[int] = None

    l_chunk = sr * 40
    # audio is [n_channels, T_total]; chunk along the time axis (last dim).
    T_total = int(audio.shape[-1])
    n_chunks_audio = (T_total + l_chunk - 1) // l_chunk

    # Write in batched transactions; per-key begin/commit is extremely slow.
    commit_bytes = 256 * 1024**2  # ~256 MiB
    bytes_in_txn = 0
    txn = env.begin(write=True)
    try:
        for i_chunk in range(int(n_chunks_audio)):
            a = audio[:, i_chunk * l_chunk : (i_chunk + 1) * l_chunk]  # [n_channels, chunk_len]
            f = features[i_chunk * l_chunk : (i_chunk + 1) * l_chunk]
            if a.shape[-1] < rf:
                continue

            # Downsample features to control frames
            feat = f.t().unsqueeze(0)  # [1, F, Ta]
            feat = torch.nn.functional.interpolate(
                feat,
                scale_factor=1 / rf,
                mode="linear",
                align_corners=False,
            ).squeeze(0).t()  # [T_ctl, F]

            # Latents (if encoder exists)
            if getattr(ddsp, "encoder", None) is not None:
                mu, scale = encoder(a.unsqueeze(0))
                if hasattr(encoder, "reparametrize"):
                    z, _ = encoder.reparametrize(mu, scale)
                elif hasattr(ddsp, "reparametrize"):
                    z, _ = ddsp.reparametrize(mu, scale)
                else:
                    z = mu
                z = ddsp._smooth_latents(z).squeeze(0)
                z = z.to(feat.device)
            else:
                Dz = int(getattr(ddsp, "latent_size", 0))
                if Dz > 0:
                    z = torch.zeros(feat.size(0), Dz, device=feat.device, dtype=feat.dtype)
                else:
                    z = torch.empty((feat.size(0), 0), device=feat.device, dtype=feat.dtype)

            # Align time dims: encoder downsampling may differ from the feature
            # downsampling by ±1, especially on the trailing partial chunk.
            T_ctl = min(feat.size(0), z.size(0))
            feat = feat[:T_ctl]
            z = z[:T_ctl]

            seq = torch.cat([feat, z], dim=-1)  # [T_ctl, D]

            T = int(seq.size(0))
            n_windows = (T - seq_len) // stride + 1
            if n_windows <= 0:
                continue

            for i in range(int(n_windows)):
                win = seq[i * stride : i * stride + seq_len]
                if win.size(0) != seq_len:
                    continue
                key = f"controls:{idx:08d}".encode()
                buf = win.detach().to("cpu").contiguous().to(torch.float32).numpy().tobytes(order="C")

                while True:
                    try:
                        txn.put(key, buf)
                        break
                    except lmdb.MapFullError:
                        txn.abort()
                        env.set_mapsize(int(env.info()["map_size"] * 1.5))
                        txn = env.begin(write=True)

                bytes_in_txn += len(key) + len(buf)
                if bytes_in_txn >= commit_bytes:
                    txn.commit()
                    txn = env.begin(write=True)
                    bytes_in_txn = 0

                if num_controls is None:
                    num_controls = int(win.size(1))
                idx += 1
    finally:
        try:
            txn.commit()
        except Exception:
            pass

    if idx == 0:
        env.close()
        raise RuntimeError("No control windows produced; check dataset length and seq_len.")

    if num_controls is None:
        env.close()
        raise RuntimeError("No control windows produced; check dataset length and seq_len.")

    meta = {
        "num_sequences": int(idx),
        "seq_len": int(seq_len),
        "num_controls": int(num_controls),
        "stride": int(stride),
        "stride_factor": float(stride_factor),
    }

    meta_buf = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)
    while True:
        try:
            with env.begin(write=True) as txn_meta:
                txn_meta.put(b"metadata", meta_buf)
            break
        except lmdb.MapFullError:
            env.set_mapsize(int(env.info()["map_size"] * 1.5))

    # Ensure metadata is flushed when using map_async.
    try:
        env.sync()
    except Exception:
        pass
    env.close()
    return {"path": out_path, **meta, "rebuilt": True}


def build_or_load_prior_cache_from_cfg(
    cfg: Any,
    *,
    control_space: ControlSpace,
    synth_configs: list[dict],
    device: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Main entrypoint: compute cache path, build if needed, return path + stats."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    fs = int(cfg.audio.fs)
    resampling_factor = int(cfg.model.resampling_factor)
    chunk_duration_s = float(cfg.audio.chunk_duration_s)
    n_signal = int(fs * chunk_duration_s)
    n_channels = int(cfg.audio.n_channels)

    dataset_path = str(cfg.data.dataset_path)

    ddsp_checkpoint_path = infer_ddsp_checkpoint_path(cfg)
    seq_len = int(cfg.prior.model.max_len)
    stride_factor = float(cfg.prior.dataset.stride_factor)

    cache_key = _create_cache_key(
        dataset_path=dataset_path,
        n_signal=n_signal,
        sampling_rate=fs,
        resampling_factor=resampling_factor,
        n_channels=n_channels,
        control_space=control_space,
        ddsp_checkpoint_path=ddsp_checkpoint_path,
        seq_len=seq_len,
        stride_factor=stride_factor,
    )

    out_path = _default_cache_path(dataset_path, cache_key)

    # Dataset (same as training)
    audio_ds = AudioFeatureDataset(
        dataset_path=dataset_path,
        n_signal=n_signal,
        sampling_rate=fs,
        resampling_factor=resampling_factor,
        control_space=control_space,
        device=dev,
    )

    # Load trained DDSP
    ddsp = DDSP.load_from_checkpoint(
        ddsp_checkpoint_path,
        strict=False,
        streaming=False,
        device=dev,
        control_space=control_space,
        synth_configs=synth_configs,
        fs=fs,
        resampling_factor=resampling_factor,
        latent_smoothing_kernel=int(cfg.model.latent_smoothing_kernel),
        decoder_gru_layers=int(cfg.model.decoder_gru_layers),
        learning_rate=float(cfg.model.learning_rate),
        plateau_patience=int(cfg.model.plateau_patience),
        capacity=int(cfg.model.capacity),
        losses=[dict(l) for l in getattr(cfg, 'losses', [])],
        adversarial_loss=bool(cfg.adversarial.enabled),
        adv_g_start_epoch=int(cfg.adversarial.schedule.g_start_epoch),
        adv_d_start_epoch=int(cfg.adversarial.schedule.d_start_epoch),
        adv_gen_weight=float(cfg.adversarial.weights.gen),
        adv_disc_weight=float(cfg.adversarial.weights.disc),
        adv_fm_weight=float(cfg.adversarial.weights.fm),
    ).to(dev)

    stats = ensure_prior_controls_lmdb(
        audio_ds=audio_ds,
        ddsp=ddsp,
        out_path=out_path,
        seq_len=seq_len,
        stride_factor=stride_factor,
        device=dev,
    )
    return out_path, stats


# ---------------------------------------------------------------------------
# Discrete-token caches (LatentCompressor codes for PriorDiscrete training)
# ---------------------------------------------------------------------------


def _default_tokens_cache_path(dataset_path: str, cache_key: str) -> str:
    base_dir = os.path.dirname(dataset_path)
    return os.path.join(base_dir, f"prior_tokens_cache_{cache_key}.lmdb")


def _compressor_signature(ckpt_path: str) -> str:
    st = os.stat(ckpt_path)
    return f"{ckpt_path}:{int(st.st_mtime)}:{int(st.st_size)}"


def _create_tokens_cache_key(
    *,
    dataset_path: str,
    n_signal: int,
    sampling_rate: int,
    resampling_factor: int,
    n_channels: int,
    control_space: ControlSpace,
    ddsp_checkpoint_path: str,
    compressor_checkpoint_path: str,
    token_seq_len: int,
    stride_factor: float,
) -> str:
    ckpt_sig = os.stat(ddsp_checkpoint_path)
    ddsp_sig = f"{ddsp_checkpoint_path}:{int(ckpt_sig.st_mtime)}:{int(ckpt_sig.st_size)}"
    comp_sig = _compressor_signature(compressor_checkpoint_path)
    field_sig = _control_space_signature(control_space)
    key_string = (
        f"{dataset_path}_{n_signal}_{sampling_rate}_{resampling_factor}_{n_channels}_"
        f"tok{int(token_seq_len)}_{stride_factor}_{field_sig}_{ddsp_sig}_{comp_sig}"
    )
    return hashlib.md5(key_string.encode()).hexdigest()[:8]


def ensure_prior_tokens_lmdb(
    *,
    audio_ds: AudioFeatureDataset,
    ddsp: DDSP,
    compressor: LatentCompressor,
    out_path: str,
    token_seq_len: int,
    stride_factor: float,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Build token LMDB cache if missing.

    LMDB schema:
      - key 'metadata' -> pickled dict
      - keys 'tokens:{idx:08d}' -> int16 bytes of shape [token_seq_len, num_codebooks]
    """
    if os.path.exists(out_path):
        env = lmdb.open(out_path, readonly=True, lock=False, readahead=True, subdir=True)
        with env.begin() as txn:
            meta_buf = txn.get(b"metadata")
        env.close()
        if meta_buf is None:
            raise RuntimeError(f"Existing token cache missing metadata: {out_path}")
        return {"path": out_path, **pickle.loads(meta_buf), "rebuilt": False}

    print("Building prior tokens LMDB cache at:", out_path)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ddsp = ddsp.to(dev)
    compressor = compressor.to(dev)
    if hasattr(ddsp, "eval"):
        ddsp.eval()
    compressor.eval()

    if getattr(compressor, "vq_enabled", False) is not True:
        raise RuntimeError("LatentCompressor checkpoint is not VQ-enabled; cannot build token cache")
    if getattr(compressor, "use_skip_connections", True) is True:
        raise RuntimeError("LatentCompressor must have use_skip_connections=False for codes-only decoding")

    encoder = getattr(ddsp, "encoder", None)
    if encoder is not None and hasattr(encoder, "streaming"):
        encoder.streaming = False

    audio = audio_ds._audio.to(dev)
    features = audio_ds._features.to(dev)  # audio-rate features
    sr = getattr(audio_ds, "_sampling_rate", None)
    if sr is None:
        sr = getattr(ddsp, "fs", 44100)
    sr = int(sr)
    rf = int(ddsp.resampling_factor)

    token_stride = max(1, int(token_seq_len * float(stride_factor)))

    # Infer VQ params from compressor.
    codebook_size = int(getattr(getattr(compressor, "hparams", None), "vq_codebook_size", 0) or getattr(compressor, "vq_codebook_size", 0) or 0)
    if codebook_size <= 0:
        codebook_size = int(getattr(getattr(compressor, "hparams", None), "vq_codebook_size", 1024))
    num_codebooks = int(getattr(compressor, "vq_num_codebooks", 1))
    strides = list(getattr(getattr(compressor, "hparams", None), "strides", []) or getattr(compressor, "strides", []))
    compression_ratio = int(getattr(compressor, "compression_ratio", 1))

    # LMDB writer
    env = lmdb.open(
        out_path,
        map_size=16 * 1024**3,
        readahead=False,
        writemap=False,
        max_dbs=1,
        sync=False,
        metasync=False,
        map_async=True,
        subdir=True,
    )

    idx = 0

    l_chunk = sr * 40
    # audio is [n_channels, T_total]; chunk along the time axis (last dim).
    T_total = int(audio.shape[-1])
    n_chunks_audio = (T_total + l_chunk - 1) // l_chunk

    commit_bytes = 256 * 1024**2
    bytes_in_txn = 0
    txn = env.begin(write=True)
    try:
        for i_chunk in range(int(n_chunks_audio)):
            a = audio[:, i_chunk * l_chunk : (i_chunk + 1) * l_chunk]  # [n_channels, chunk_len]
            f = features[i_chunk * l_chunk : (i_chunk + 1) * l_chunk]
            if a.shape[-1] < rf:
                continue

            # Downsample features to control frames
            feat = f.t().unsqueeze(0)  # [1, F, Ta]
            feat = torch.nn.functional.interpolate(
                feat,
                scale_factor=1 / rf,
                mode="linear",
                align_corners=False,
            ).squeeze(0).t()  # [T_ctl, F]

            # Latents (if encoder exists). Encoder downmixes the [1, n_channels, T] input internally.
            if getattr(ddsp, "encoder", None) is not None:
                mu, scale = encoder(a.unsqueeze(0))
                if hasattr(encoder, "reparametrize"):
                    z, _ = encoder.reparametrize(mu, scale)
                elif hasattr(ddsp, "reparametrize"):
                    z, _ = ddsp.reparametrize(mu, scale)
                else:
                    z = mu
                z = ddsp._smooth_latents(z).squeeze(0)
                z = z.to(feat.device)
            else:
                Dz = int(getattr(ddsp, "latent_size", 0))
                if Dz > 0:
                    z = torch.zeros(feat.size(0), Dz, device=feat.device, dtype=feat.dtype)
                else:
                    z = torch.empty((feat.size(0), 0), device=feat.device, dtype=feat.dtype)

            # Align time dims: encoder downsampling may differ from the feature
            # downsampling by ±1, especially on the trailing partial chunk.
            T_ctl = min(feat.size(0), z.size(0))
            feat = feat[:T_ctl]
            z = z[:T_ctl]

            controls = torch.cat([feat, z], dim=-1)  # [T_ctl, D]

            with torch.no_grad():
                tok = compressor.encode_codes(controls.unsqueeze(0)).squeeze(0)
            if tok.dim() == 1:
                tok = tok.unsqueeze(-1)

            T_low = int(tok.size(0))
            n_windows = (T_low - token_seq_len) // token_stride + 1
            if n_windows <= 0:
                continue

            for i in range(int(n_windows)):
                win = tok[i * token_stride : i * token_stride + token_seq_len]
                if win.size(0) != token_seq_len:
                    continue
                key = f"tokens:{idx:08d}".encode()
                buf = win.detach().to("cpu").contiguous().to(torch.int16).numpy().tobytes(order="C")

                while True:
                    try:
                        txn.put(key, buf)
                        break
                    except lmdb.MapFullError:
                        txn.abort()
                        env.set_mapsize(int(env.info()["map_size"] * 1.5))
                        txn = env.begin(write=True)

                bytes_in_txn += len(key) + len(buf)
                if bytes_in_txn >= commit_bytes:
                    txn.commit()
                    txn = env.begin(write=True)
                    bytes_in_txn = 0

                idx += 1
    finally:
        try:
            txn.commit()
        except Exception:
            pass

    if idx == 0:
        env.close()
        raise RuntimeError("No token windows produced; check dataset length and token_seq_len.")

    control_rate_hz = float(sr) / float(rf)
    steps_per_second = control_rate_hz / float(compression_ratio)

    meta = {
        "num_sequences": int(idx),
        "seq_len": int(token_seq_len),
        "num_codebooks": int(num_codebooks),
        "codebook_size": int(codebook_size),
        "stride": int(token_stride),
        "stride_factor": float(stride_factor),
        "compression_ratio": int(compression_ratio),
        "strides": strides,
        "control_rate_hz": float(control_rate_hz),
        "steps_per_second": float(steps_per_second),
    }

    meta_buf = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)
    while True:
        try:
            with env.begin(write=True) as txn_meta:
                txn_meta.put(b"metadata", meta_buf)
            break
        except lmdb.MapFullError:
            env.set_mapsize(int(env.info()["map_size"] * 1.5))

    try:
        env.sync()
    except Exception:
        pass
    env.close()
    return {"path": out_path, **meta, "rebuilt": True}


def build_or_load_prior_tokens_cache_from_cfg(
    cfg: Any,
    *,
    control_space: ControlSpace,
    synth_configs: list[dict],
    device: Optional[str] = None,
    compressor_ckpt: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    fs = int(cfg.audio.fs)
    resampling_factor = int(cfg.model.resampling_factor)
    chunk_duration_s = float(cfg.audio.chunk_duration_s)
    n_signal = int(fs * chunk_duration_s)
    n_channels = int(cfg.audio.n_channels)

    dataset_path = str(cfg.data.dataset_path)

    ddsp_checkpoint_path = infer_ddsp_checkpoint_path(cfg)

    # Token sequence length in low-rate steps.
    token_seq_len = int(cfg.prior.model.max_len)
    if getattr(cfg.prior, "discrete", None) is not None:
        token_seq_len = int(getattr(getattr(cfg.prior.discrete, "model", {}), "max_len", token_seq_len))

    stride_factor = float(getattr(cfg.prior.dataset, "stride_factor", 0.2))
    if getattr(cfg.prior, "discrete", None) is not None and getattr(cfg.prior.discrete, "dataset", None) is not None:
        stride_factor = float(getattr(cfg.prior.discrete.dataset, "stride_factor", stride_factor))

    if compressor_ckpt is None and getattr(cfg.prior, "discrete", None) is not None:
        compressor_ckpt = getattr(cfg.prior.discrete, "compressor_ckpt", None)
    if compressor_ckpt is None:
        raise RuntimeError("A compressor checkpoint is required to build the token cache")
    compressor_ckpt = str(compressor_ckpt)

    cache_key = _create_tokens_cache_key(
        dataset_path=dataset_path,
        n_signal=n_signal,
        sampling_rate=fs,
        resampling_factor=resampling_factor,
        n_channels=n_channels,
        control_space=control_space,
        ddsp_checkpoint_path=ddsp_checkpoint_path,
        compressor_checkpoint_path=compressor_ckpt,
        token_seq_len=token_seq_len,
        stride_factor=stride_factor,
    )
    out_path = _default_tokens_cache_path(dataset_path, cache_key)

    audio_ds = AudioFeatureDataset(
        dataset_path=dataset_path,
        n_signal=n_signal,
        sampling_rate=fs,
        resampling_factor=resampling_factor,
        control_space=control_space,
        device=dev,
    )

    ddsp = DDSP.load_from_checkpoint(
        ddsp_checkpoint_path,
        strict=False,
        streaming=False,
        device=dev,
        control_space=control_space,
        synth_configs=synth_configs,
        fs=fs,
        resampling_factor=resampling_factor,
        latent_smoothing_kernel=int(cfg.model.latent_smoothing_kernel),
        decoder_gru_layers=int(cfg.model.decoder_gru_layers),
        learning_rate=float(cfg.model.learning_rate),
        plateau_patience=int(cfg.model.plateau_patience),
        capacity=int(cfg.model.capacity),
        losses=[dict(l) for l in getattr(cfg, "losses", [])],
        adversarial_loss=bool(cfg.adversarial.enabled),
        adv_g_start_epoch=int(cfg.adversarial.schedule.g_start_epoch),
        adv_d_start_epoch=int(cfg.adversarial.schedule.d_start_epoch),
        adv_gen_weight=float(cfg.adversarial.weights.gen),
        adv_disc_weight=float(cfg.adversarial.weights.disc),
        adv_fm_weight=float(cfg.adversarial.weights.fm),
    ).to(dev)

    compressor = LatentCompressor.load_from_checkpoint(compressor_ckpt).to(dev)

    stats = ensure_prior_tokens_lmdb(
        audio_ds=audio_ds,
        ddsp=ddsp,
        compressor=compressor,
        out_path=out_path,
        token_seq_len=token_seq_len,
        stride_factor=stride_factor,
        device=dev,
    )
    return out_path, stats
