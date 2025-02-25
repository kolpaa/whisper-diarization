<h1 align="center">Speaker Diarization Using OpenAI Whisper</h1>


<p align="center">
  <a href="https://github.com/MahmoudAshraf97/whisper-diarization/actions/workflows/test_run.yml">
    <img src="https://github.com/MahmoudAshraf97/whisper-diarization/actions/workflows/test_run.yml/badge.svg"
         alt="Build Status">
  </a>
  <a href="https://github.com/MahmoudAshraf97/whisper-diarization/stargazers">
    <img src="https://img.shields.io/github/stars/MahmoudAshraf97/whisper-diarization.svg?colorA=orange&colorB=orange&logo=github"
         alt="GitHub stars">
  </a>
  <a href="https://github.com/MahmoudAshraf97/whisper-diarization/issues">
        <img src="https://img.shields.io/github/issues/MahmoudAshraf97/whisper-diarization.svg"
             alt="GitHub issues">
  </a>
  <a href="https://github.com/MahmoudAshraf97/whisper-diarization/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/MahmoudAshraf97/whisper-diarization.svg"
             alt="GitHub license">
  </a>
  <a href="https://twitter.com/intent/tweet?text=&url=https%3A%2F%2Fgithub.com%2FMahmoudAshraf97%2Fwhisper-diarization">
  <img src="https://img.shields.io/twitter/url/https/github.com/MahmoudAshraf97/whisper-diarization.svg?style=social" alt="Twitter">
  </a> 
  </a>
  <a href="https://colab.research.google.com/github/MahmoudAshraf97/whisper-diarization/blob/main/Whisper_Transcription_%2B_NeMo_Diarization.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>
 
</p>


# 
Speaker Diarization pipeline based on OpenAI Whisper

<img src="https://github.blog/wp-content/uploads/2020/09/github-stars-logo_Color.png" alt="drawing" width="25"/> **Please, star the project on github (see top-right corner) if you appreciate my contribution to the community!**

## What is it
Python 3.12.2

Install ffmpeg from gyan.dev. Then add the folder where ffmpeg.exe is located (e.g. C:\ffmpeg\bin) to the system variables PATH.
https://www.gyan.dev/ffmpeg/builds/

Install perl. It's a programming language like python. Strawberry perl worked fine for me.
https://strawberryperl.com/

Install Python 3.10.11. You can use later Python versions (e.g. 3.11, 3.12, etc), but 3.10.11 has given me no issues. There are later versions of 3.10 (e.g. 3.10.12) but there aren't binaries for windows and you have to build them yourself.
Create a Python virtual environment for your project. It's a good way to keep Python versions and dependencies separate for each project. To do that go to the root folder where you want to create the virtual environment, e.g. C:\Users\Bill\PurePython_envs. Open cmd at that location (or cd there) and run this command to create the Python virtual environment with venv: "C:\Program Files\Python310\python.exe" -m venv <your_venv_name>. The first part of this command specifies which Python version we want to use for this venv. Then cd into the venv folder by running cd "C:\Users\Bill\PurePython_envs\<your_venv_name>" and activate the venv by running scripts\activate. This command needs to be run from the environment folder (C:\Users\Bill\PurePython_envs\<your_venv_name> in this case). You're already in that folder at this point in the instructions but I'm including it so you know how to activate it in the future. If the venv has been activated correctly you should see the name of the venv before the location in cmd, e.g. (<your_venv_name>) C:\Users\Bill\PurePython_envs\<your_venv_name>.

Upgrade pip, setuptools and wheel by running: python -m pip install --upgrade pip setuptools wheel. I'm adding python -m before this command to make sure I'm using the correct Python environment.

Install CUDA Toolkit 12.4 and cuDNN (latest version is fine). There are newer versions of CUDA Toolkit but the latest version of pytorch (which is what we're using to GPU accelerate transcription) works with CUDA up to 12.4.

Install pytorch with CUDA 12.4 support. You can see the instructions for yourself here, but this is the command: python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Install cython by running: python -m pip install cython

Download the zip of the entire whisper-diarization project and extract it somewhere, e.g. C:\Users\Bill\VSC_WorkingDirectory\whisper-diarization. cd into that folder from the same cmd terminal that has the venv active and we have been using for all the previous steps. Run this command to install whisper-diarization: C:\Users\Bill\PurePython_envs\<your_venv_name>\Scripts\python.exe -m pip install -c constraints.txt -r requirements.txt
Open the project folder (C:\Users\Bill\VSC_WorkingDirectory\whisper-diarization in this case) in VS Code and select the Python interpreter (by typing >Python: Select Interpreter in the command palette) of the venv we've created (in this case C:\Users\Bill\PurePython_envs\<your_venv_name>\scripts\python.exe).

Prepare your audio files by making sure that they're in .wav format, 16 KHz, mono (that's the format that whisper prefers (whisper is doing the transcription)). Audacity is a great free tool for that. Place the audio file in the project folder.
Finally to run the script open a terminal in the VS Code window we opened earlier and run: python diarize.py -a "your_audio_file.wav" --no-stem --whisper-model large-v3 --language en --device cuda. The resulting diarized transcriptions will appear as .txt and .srt files in the same location.

    

```
## Usage 

```
python diarize.py -a AUDIO_FILE_NAME
```

If your system has enough VRAM (>=10GB), you can use `diarize_parallel.py` instead, the difference is that it runs NeMo in parallel with Whisper, this can be beneficial in some cases and the result is the same since the two models are nondependent on each other. This is still experimental, so expect errors and sharp edges. Your feedback is welcome.

## Command Line Options

- `-a AUDIO_FILE_NAME`: The name of the audio file to be processed
- `--no-stem`: Disables source separation
- `--whisper-model`: The model to be used for ASR, default is `medium.en`
- `--suppress_numerals`: Transcribes numbers in their pronounced letters instead of digits, improves alignment accuracy
- `--device`: Choose which device to use, defaults to "cuda" if available
- `--language`: Manually select language, useful if language detection failed
- `--batch-size`: Batch size for batched inference, reduce if you run out of memory, set to 0 for non-batched inference

## Known Limitations
- Overlapping speakers are yet to be addressed, a possible approach would be to separate the audio file and isolate only one speaker, then feed it into the pipeline but this will need much more computation
- There might be some errors, please raise an issue if you encounter any.

## Future Improvements
- Implement a maximum length per sentence for SRT

## Acknowledgements
Special Thanks for [@adamjonas](https://github.com/adamjonas) for supporting this project
This work is based on [OpenAI's Whisper](https://github.com/openai/whisper) , [Faster Whisper](https://github.com/guillaumekln/faster-whisper) , [Nvidia NeMo](https://github.com/NVIDIA/NeMo) , and [Facebook's Demucs](https://github.com/facebookresearch/demucs)

## Citation
If you use this in your research, please cite the project:

```bibtex
@unpublished{hassouna2024whisperdiarization,
  title={Whisper Diarization: Speaker Diarization Using OpenAI Whisper},
  author={Ashraf, Mahmoud},
  year={2024}
}
```
