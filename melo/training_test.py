import subprocess
import sys
import time
import os

def run_training_loop(config_path, num_gpus, master_port=10902):
    """
    Menjalankan loop pelatihan secara otomatis me-resume.
    Logika ini meniru script Bash dengan menjalankan torchrun
    dan membunuh proses yang tersisa jika terjadi crash.
    """
    if not config_path or not num_gpus:
        print("Error: CONFIG_PATH dan NUM_GPUS harus disediakan.")
        return

    # 1. Menentukan MODEL_NAME
    # Setara dengan MODEL_NAME=$(basename "$(dirname $CONFIG)") di Bash
    try:
        # Mendapatkan nama direktori induk, lalu mengambil nama dasarnya
        model_name = os.path.basename(os.path.dirname(config_path))
    except Exception:
        # Fallback jika ada masalah dalam mendapatkan nama direktori
        print("Warning: Gagal mendapatkan MODEL_NAME dari path config. Menggunakan 'default_model'.")
        model_name = "default_model"

    # Perintah utama torchrun
    torchrun_command = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        f"--master_port={master_port}",
        "train.py",
        "--c", config_path,
        "--model", model_name
    ]

    while True:
        print("-" * 50)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Memulai pelatihan...")
        print(f"Perintah: {' '.join(torchrun_command)}")
        
        # 2. Menjalankan perintah torchrun
        # Setara dengan `torchrun ... train.py ...` di Bash
        try:
            # Gunakan subprocess.run untuk menjalankan perintah, ini akan menunggu sampai selesai/crash
            process = subprocess.run(
                torchrun_command,
                check=False,  # Jangan raise error pada non-zero exit code (kita mengharapkan crash)
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            
            # Jika keluar dengan sukses (exit code 0), loop akan dihentikan
            if process.returncode == 0:
                print("Pelatihan selesai dengan sukses!")
                break
            else:
                print(f"Pelatihan crash (Exit Code: {process.returncode}). Mencoba me-resume...")

        except FileNotFoundError:
            print("Error: torchrun atau train.py tidak ditemukan. Pastikan sudah terinstal dan ada di PATH.")
            break
        except Exception as e:
            print(f"Terjadi Error tak terduga: {e}. Mencoba me-resume...")

        # 3. Membersihkan proses yang tersisa (auto-resume logic)
        # Setara dengan `for PID in $(ps -aux | grep $CONFIG | grep python | awk '{print $2}') ... kill -9 $PID`
        print("Mencari dan membunuh proses Python/training yang tersisa...")
        
        # Perintah ps/grep/awk untuk mendapatkan PID
        # Kita perlu menjalankan ini di shell untuk piping
        cleanup_command = (
            f"ps -aux | grep {config_path} | grep python | awk '{{print $2}}'"
        )

        try:
            # Jalankan perintah untuk mendapatkan daftar PID
            pids_output = subprocess.run(
                cleanup_command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            ).stdout.strip()
            
            # Pisahkan PID dan bunuh
            pids_to_kill = [p for p in pids_output.split('\n') if p]
            
            if pids_to_kill:
                print(f"Ditemukan PID yang tersisa: {', '.join(pids_to_kill)}")
                for pid in pids_to_kill:
                    try:
                        # Bunuh proses
                        os.kill(int(pid), 9) # 9 adalah SIGKILL (-9)
                        print(f"Killed PID {pid}")
                    except ProcessLookupError:
                        # Proses sudah tidak ada
                        pass 
                    except Exception as e:
                        print(f"Gagal membunuh PID {pid}: {e}")
            else:
                print("Tidak ditemukan proses tersisa untuk dibersihkan.")

        except subprocess.CalledProcessError as e:
            # Ini mungkin terjadi jika grep tidak menemukan apa-apa, yang tidak masalah.
            print(f"Warning saat mencari proses tersisa: {e.stderr.strip()}")
        except Exception as e:
            print(f"Error saat mencoba membersihkan proses: {e}")

        # 4. Jeda sebelum restart
        # Setara dengan `sleep 30` di Bash
        print(f"Menunggu 30 detik sebelum memulai kembali...")
        time.sleep(30)
        
        
if __name__ == "__main__":
    # Memeriksa argumen dari command line (seperti $1 dan $2 di Bash)
    if len(sys.argv) < 3:
        print("Penggunaan: python script_name.py <config_path> <num_gpus>")
        print("Contoh: python script_name.py configs/my_model/config.yaml 4")
        sys.exit(1)

    CONFIG = sys.argv[1]
    GPUS = sys.argv[2]
    
    # Menjalankan fungsi utama
    # PORT tetap 10902 seperti di script Bash
    run_training_loop(CONFIG, GPUS)