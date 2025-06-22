
<br/>
<div align="center">
<a href="https://github.com/amarcelq/dapi_amphibien/">
<img src="docs/pics/frog_symbol.png" alt="Logo" width="80" height="80">
</a>
<h3 align="center">Frogs</h3>
<p align="center">
A web application to discover different frogs in a recording
<br/>
<br/>
<a href="#getting-started"><strong>Getting Started »</strong></a>
<a href="#how-it-works-and-what-weve-tried"><strong>How it works »</strong></a>
<a href="#future-steps"><strong>Future Steps »</strong></a>

  


</p>
</div>

## About The Project

![Product Screenshot](docs/pics/app.png)

This project is a research project, trying to separate unique frog calls from one recording. 
### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

- [Django](https://www.djangoproject.com)
- [Docker](https://www.docker.com)
- [wavesurfer.js](https://wavesurfer.xyz)
- [Celery](https://docs.celeryq.dev/en/stable/)
- [Redis](https://redis.io)
- [FastAPI](https://fastapi.tiangolo.com)
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.
### Prerequisites

You need a working docker deamon installed and running. Confirm with `docker -v`.

In some cases it could be necessary to install the dependencies manually (e.g. when developing).

- Django Requirements
  ```sh
  uv sync
  ```
- JS/CSS Requirements
  ```sh
  cd assets
  yarn
  ```
- Audio Processing Requirements
  ```sh
  cd audio_processing
  uv sync
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/amarcelq/dapi_amphibien.git
   ```
2. Copy the `.env.example` file into `.env`. You dont _have_ to change something in it, but you _should_ change things like DB passwords and the secure keys.
   ```sh
   cp .env.example .env
   ```
3. Start everything using docker compose
   ```sh
   docker compose up -d --build
   ```
4. If its the first time starting it, you have to initialise the Database and Django schemas. Run the following commands:
   ```sh
   ./run manage makemigrations
   ./run manage migrate
5. Now everything should be up and running and you should see the application at `localhost:8000`!
## Usage

When the page is running, drag n' drop a sound file (`.wav`) onto the green box. The site will begin processing it showing a progress bar. 
![Process Image](docs/pics/progress.png)
After it is done progressing, it will show the originial recording, as well as all found clusters of sound. Those should be frogs, but it can also happen that other sounds find their way there. Just Click on the wave forms to listen to a sample of them. To see all samples of one cluster, exapnd the tile by clicking on "Show more". 
![Tiles Image]()

## How It Works
> Note: This program is designed to function with or without the website interface. Refer to `audio_processing/main.py` for core logic. This README focuses on the website implementation.
1. Receive the input frog mixture from the web interface.  
2. Apply denoising and optional preprocessing steps such as trimming and converting to mono.  
3. Perform sound separation using breadth-first search with ConvTasNet, generating two splits. Evaluate each split’s "frogginess" by measuring its Euclidean distance to the mean froggy vector versus the mean non-froggy vector (means computed from multiple frog recordings and random non-frog sounds; see `audio_processing/data/utils.py`).  
   - If froggy, expand the branch until max depth is reached.  
   - If not froggy, terminate the branch.  
   Pretrained PyTorch weights are fine-tuned to distinguish anuran calls from other sounds. Training data includes Kaggle’s Anuran dataset mixed with various pond, church bell, and car sounds from YouTube (see `audio_processing/data/utils.py`). 
4. Segment the audio into non-silent parts.  
5. Save each segment as a WAV file and display them on the website front end.

## Problems and Solutions
- **Environmental and stationary noise**: Microphones capture both stationary and ambient noise (e.g., crickets, white noise). Use a **Spectral Gate** for denoising—this produced the best results in our tests. Provide a representative noise sample to the denoiser (see `audio_processing/preprocessing/denoise.py`).
- **Training a custom separator is not feasible**: Due to hardware constraints, use a **pretrained Conv-TasNet**. Alternatives like MixIT (Google), Wave-U-Net, NMF, and FastICA were tested; MixIT required excessive implementation effort, and the others underperformed relative to Conv-TasNet.
- **Conv-TasNet training is too computationally expensive**: Mitigate this by splitting audio files into **8-second segments** and using a **batch size of 4**.
- **Conv-TasNet is designed for two-speaker separation**: Apply **breadth-first search (BFS)** to recursively split outputs until a maximum depth is reached (see `How it works` for implementation details).
- **Inconsistent output volume**: Normalize all resulting audio snippets to a **consistent LUFS level**.

## Future Steps
Here now follows a list with things that could be added to the project or which's foundations already been laid, but would be outside of the scope for this project.

- User Accounts:
  - There could be user accounts to store found clusters, share them with other useres etc. The foundation for that is already there, but wa snot necessary for this project.
- Uploading multiple files:
  - Right now only one file can be uplaoded and will be analyzed. In the future the user could upload multiple files which could be joined internaly to process all at once.
- A viable strategy involves training ConvTasNet to distinguish between froggy and non-froggy sounds, as detailed in the `How it works` section and the `audio_processing/data/utils.py` module. Subsequently, features can be extracted—potentially using OpenL3—and dimensionality reduction applied, for instance via PCA, followed by clustering. Although mapping from the reduced feature space back to the original audio files presents significant challenges, this approach may yield valuable insights. The principal limitation lies in ConvTasNet’s computational demands, exacerbated by the high dimensionality of audio data, which exceeds the available processing resources.
## License

Distributed under the MIT License. See [MIT License](https://opensource.org/licenses/MIT) for more information.
## Acknowledgments

Here are some ressources we used to create this project. Make sure to check them out!

- [makeread.me](https://github.com/ShaanCoding/ReadME-Generator)
- [othneildrew](https://github.com/othneildrew/Best-README-Template)
- [Django Docker Template](https://github.com/nickjj/docker-django-example)