import warnings
import os
from gfpgan import GFPGANer
import torch

warnings.filterwarnings("ignore")

# Dapatkan path absolut dari direktori tempat skrip ini berada.
# Ini memastikan path akan selalu benar, tidak peduli dari mana skrip ini dipanggil.
script_dir = os.path.dirname(os.path.abspath(__file__))

def load_sr():
    """
    Memuat model GFPGAN dengan path absolut ke checkpoint dan memeriksa perangkatnya.
    """
    # Bangun path absolut ke file model GFPGAN.
    model_path_abs = os.path.join(script_dir, "checkpoints", "GFPGANv1.4.pth")

    # Tambahkan pemeriksaan untuk memastikan file ada, untuk debugging yang lebih mudah.
    if not os.path.exists(model_path_abs):
        raise FileNotFoundError(
            f"Model checkpoint GFPGAN tidak ditemukan di path yang diharapkan: {model_path_abs}"
        )
    
    print(f"Memuat model GFPGAN dari: {model_path_abs}")

    # GFPGANer akan secara otomatis menggunakan GPU jika torch.cuda.is_available() adalah True.
    run_params = GFPGANer(
        model_path=model_path_abs,  # Gunakan path absolut
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )

    # --- BLOK PENGECEKAN GPU ---
    # Cek perangkat tempat model GFPGAN dimuat setelah inisialisasi.
    try:
        # Mengakses model gfpgan di dalam objek run_params dan memeriksa perangkat parameternya.
        gfpgan_device = next(run_params.gfpgan.parameters()).device
        print(f"-> Model GFPGAN berhasil dimuat dan berjalan di perangkat: {str(gfpgan_device).upper()}")
        if 'cpu' in str(gfpgan_device):
             print("-> PERINGATAN: GFPGAN berjalan di CPU, proses 'Enhanced' akan lambat.")
    except Exception as e:
        print(f"-> Tidak dapat memverifikasi perangkat GFPGAN: {e}")
    # --- AKHIR BLOK PENGECEKAN ---

    return run_params


def upscale(image, properties):
    """
    Meningkatkan kualitas gambar (upscale) menggunakan model yang telah dimuat.
    """
    _, _, output = properties.enhance(
        image, has_aligned=False, only_center_face=False, paste_back=True
    )
    return output
