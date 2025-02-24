import os
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import subprocess
from tkinter import ttk
import glob
from docx import Document
import threading
import sys
import psutil
from multiprocessing import Process
from multiprocessing import Pool, cpu_count
from functools import partial
import time


global process
root = None
process = None
pause = False

def convert_to_wav(folder_path: str, output_folder: str = None, target_format: str = "wav"):
    if output_folder is None:
        output_folder = folder_path
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
        
        file_name, file_ext = os.path.splitext(filename)
        if file_ext.lower() in [".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a", ".opus"]:
            output_path = os.path.join(output_folder, f"{file_name}.{target_format}")
            subprocess.run(["ffmpeg", "-i", file_path, output_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log_text.insert(tk.END, f"Конвертирован: {file_path} -> {output_path}")
            log_text.see(tk.END)

def read_stream(stream, stream_name):
    for line in iter(stream.readline, ''):
        sys.stdout.write(f"[{stream_name}] {line}") 
        log_text.insert(tk.END, line)
        log_text.see(tk.END)
        sys.stdout.flush() 
    stream.close()

def delete_files(folder_path: str):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.lower().endswith(".wav"):
            os.remove(file_path)
            log_text.insert(tk.END, f"Удалён: {file_path}")
            log_text.see(tk.END)
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.lower().endswith(".txt"):
            os.remove(file_path)
            log_text.insert(tk.END, f"Удалён: {file_path}")
            log_text.see(tk.END)
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.lower().endswith(".srt"):
            os.remove(file_path)
            log_text.insert(tk.END, f"Удалён: {file_path}")
            log_text.see(tk.END)


def txt_to_word(txt_file_path, word_file_path):
    try:
        doc = Document()
        
        with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
            lines = txt_file.readlines()
        
        for line in lines:
            doc.add_paragraph(line.strip())
        
        doc.save(word_file_path)
        log_text.insert(tk.END, f"Файл успешно сохранен как {word_file_path}")
        log_text.see(tk.END)
    except Exception as e:
        log_text.insert(tk.END, f"Произошла ошибка: {e}")
        log_text.see(tk.END)


def process_file(filename, device, quality):
    command = ["python", "whisper-diarization/diarize.py", "-a", filename, "--no-stem", "--whisper-model", quality, "--device", device]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    process.communicate() 


def transcribe_audio(source, use_gpu, quality, num_cores, youtube_folder):
    if "https://www.youtube.com" in source:
        command = ["yt-dlp", "-P", youtube_folder, "--download-archive", youtube_folder + "/downloaded.txt", "--no-post-overwrites", source, "-x", "--audio-format", "wav", "--cookies-from-browser", "firefox", "--extractor-args", "youtube:player_client=default,-web_creator"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,  bufsize=1)


        stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, "STDOUT"))
        stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, "STDERR"))

        stdout_thread.start()
        stderr_thread.start()

        # Ждем завершения процесса
        process.wait()

        # Ждем завершения потоков
        stdout_thread.join()
        stderr_thread.join()
    
    else:
        convert_to_wav(source)
    device = "cuda" if use_gpu else "cpu"
    if youtube_folder:
        source = youtube_folder

    if num_cores != 1 and num_cores != "":
        files = glob.glob(os.path.join(source, '*.wav'))
        with Pool(processes=int(num_cores)) as pool: 
            pool.map(partial(process_file, device=device, quality=quality), files)
        for filename in glob.glob(os.path.join(source, '*.wav')):
            file_audio_name = filename[:filename.rfind('.')] if '.' in filename else filename
            file_path = f"{file_audio_name}.txt"
            word_file_path = f"{file_audio_name}.docx"
            txt_to_word(file_path, word_file_path)
    else:
        for filename in glob.glob(os.path.join(source, '*.wav')):
            global pause
            while pause:
                root.update()  
                time.sleep(0.1)  
                    
            current_file_label.config(text=f"Текущий файл: {filename}")
            command = ["python", "whisper-diarization/diarize.py", "-a", filename, "--no-stem", "--whisper-model", quality, "--device", device]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,  bufsize=1)


            stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, "STDOUT"))
            stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, "STDERR"))

            stdout_thread.start()
            stderr_thread.start()


            process.wait()


            stdout_thread.join()
            stderr_thread.join()

            file_audio_name = filename[:filename.rfind('.')] if '.' in filename else filename
            file_path = f"{file_audio_name}.txt"
            word_file_path = f"{file_audio_name}.docx"
            txt_to_word(file_path, word_file_path)
        

    delete_files(source)
    
    

def select_folder(entry):
    folder = filedialog.askdirectory(title="Выберите папку")
    if folder:
        entry.delete(0, tk.END)
        entry.insert(0, folder)

def start_transcription():
    global pause
    pause = False
    source = source_entry.get()
    if not source:
        messagebox.showerror("Ошибка", "Выберите источник аудио.")
        return
    use_gpu = gpu_var.get()
    quality = quality_var.get()
    num_cores = cores_var.get()
    youtube_folder = download_entry.get()
    threading.Thread(target=transcribe_audio, args=(source, use_gpu, quality, num_cores, youtube_folder), daemon=True).start()

def pause_transcription():
    global pause
    pause = True

def resume_transcription():
    global pause
    pause = False

def main():
    global source_entry, download_entry, gpu_var, quality_var, cores_var, log_text, current_file_label
    global root
    root = tk.Tk()
    root.title("Транскрибация аудио")
    root.geometry("700x550")
    root.configure(bg='#f7f7f7')
    
    frame = ttk.LabelFrame(root, text="Настройки", padding=15)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    ttk.Label(frame, text="Источник аудио:").grid(row=0, column=0, pady=5, sticky='w')
    source_entry = ttk.Entry(frame, width=55)
    source_entry.grid(row=0, column=1, pady=5, padx=5)
    ttk.Button(frame, text="📂", command=lambda: select_folder(source_entry)).grid(row=0, column=2, padx=5)
    
    ttk.Label(frame, text="Папка для YouTube:").grid(row=1, column=0, pady=5, sticky='w')
    download_entry = ttk.Entry(frame, width=55)
    download_entry.grid(row=1, column=1, pady=5, padx=5)
    ttk.Button(frame, text="📂", command=lambda: select_folder(download_entry)).grid(row=1, column=2, padx=5)
    
    ttk.Label(frame, text="Качество:").grid(row=2, column=0, pady=5, sticky='w')
    quality_var = ttk.Combobox(frame, values=["small", "medium", "large-v3"], state='readonly', width=15)
    quality_var.grid(row=2, column=1, pady=5, padx=5, sticky='w')
    quality_var.current(0)
    
    ttk.Label(frame, text="Ядра CPU:").grid(row=3, column=0, pady=5, sticky='w')
    cores_var = ttk.Spinbox(frame, from_=1, to=os.cpu_count(), width=5, textvariable=tk.IntVar(value=1))
    cores_var.grid(row=3, column=1, pady=5, padx=5, sticky='w')
    
    gpu_var = tk.BooleanVar()
    ttk.Checkbutton(frame, text="Использовать GPU", variable=gpu_var).grid(row=4, column=0, pady=5)
    
    button_frame = ttk.Frame(root)
    button_frame.pack(pady=10)
    
    ttk.Button(button_frame, text="▶ Запуск", command=start_transcription).grid(row=0, column=0, padx=10)
    ttk.Button(button_frame, text="⏸ Пауза", command=pause_transcription).grid(row=0, column=1, padx=10)
    ttk.Button(button_frame, text="▶ Продолжить", command=resume_transcription).grid(row=0, column=2, padx=10)
    
    current_file_label = ttk.Label(root, text="Текущий файл: -", font=("Arial", 10, "bold"))
    current_file_label.pack(pady=5)
    
    log_frame = ttk.LabelFrame(root, text="Логи", padding=5)
    log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, bg="#ffffff", relief=tk.SUNKEN, borderwidth=2, font=("Courier", 10))
    log_text.pack(fill=tk.BOTH, expand=True)
    
    root.mainloop()

if __name__ == "__main__":
    main()
