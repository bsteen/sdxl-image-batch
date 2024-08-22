import glob
import multiprocessing
import os
import pickle
import random
import uuid
from PIL import Image
from tqdm import tqdm
# OTHER IMPORTS USED: torch diffusers compel accelerate omegaconf xformers peft (used for combining LoRAs)

class SDXLModel():
    def __init__(self, lora_dicts=[]):
        self.model_dir = f"{os.getcwd()}{os.sep}models{os.sep}"
        self.output_dir = f"{os.getcwd()}{os.sep}outputs{os.sep}"

        os.environ["HF_HOME"] = f"{self.model_dir}huggingface"  # Prevents `diffusors` import from storing junk in ~/.cache/huggingface
        # `diffusors` and `torch` have to be imported after "HF_HOME" is set, otherwise above line will have no effect
        # `diffusors` has to be imported in a separate process, or else it will never let go of GPU memory when trying to switch models (BUG: out of memory crash)
        os.environ["HF_DATASETS_OFFLINE"] = "1"   # Offline mode: do not fetch missing models automatically
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.lora_paths = []
        self.adapter_names = []
        self.lora_scales = []
        for lora in lora_dicts:
            path = f"{self.model_dir}lora{os.sep}{lora['model']}"
            if not path.endswith(".safetensors"):    # Assume safetensors file for LoRAs
                path += ".safetensors"
            self.lora_paths.append(path)
            self.adapter_names.append(os.path.basename(path).split(".")[0])
            self.lora_scales.append(lora["scale"])

        self.model_pipe = None

    def load_model(self, diffuser_pipe, load_msg):
        if self.model_pipe is not None:
            return
        print(load_msg, self.model_dir)

        import torch
        self.model_pipe = diffuser_pipe.from_pretrained(self.model_dir,
                                                        torch_dtype=torch.float16,  # Why is this still needed when `*.fp16.safetensors` are loaded?
                                                        add_watermarker=False,
                                                        use_safetensors=True,
                                                        local_files_only=True,
                                                        variant="fp16")             # Won't load `*.fp16.safetensors` unless this is set
        self.model_pipe.safety_checker = None

        # Load LoRAs
        if self.adapter_names:
            for path, name in tqdm(zip(self.lora_paths, self.adapter_names), total=len(self.lora_paths), desc="Loading LoRAs..."):
                assert os.path.isfile(path), f"LoRA not found: {path}"
                self.model_pipe.load_lora_weights(path, adapter_name=name, use_safetensors=True, weight_name=path.split(os.sep)[-1])

            self.model_pipe.set_adapters(self.adapter_names, adapter_weights=self.lora_scales)
            print("Fusing LoRAs...")
            self.model_pipe.fuse_lora()     # fuses LoRAs to UNet and text encoder; can speed-up inference steps and lower VRAM usage
        print("Active adapters:", self.model_pipe.get_active_adapters())

        # self.model_pipe.to("cuda")               # FIXME CUDA OOM error, try setting max_split_size_mb to avoid fragmentation, see documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
                                                   # Tried doing this at start and still OOM: os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"
                                                   # On certain batch sizes, it seems to work, but then always fails when offloading the final result from GPU to CPU
        self.model_pipe.enable_model_cpu_offload() # When limited by GPU VRAM, enable cpu offloading instead of using `.to("cuda")`

        # Enable memory efficiency settings
        if "svd" in self.model_dir:
            self.model_pipe.unet.enable_forward_chunking()               # The feed-forward layer runs in a loop instead of running with a single huge feed-forward batch size
            self.model_pipe.enable_xformers_memory_efficient_attention() # FIXME CUDA configuration error
        else:
            self.model_pipe.enable_vae_slicing()                            # Helps prevent CUDA out of memory errors when offloading multiple images from the GPU
            self.model_pipe.enable_xformers_memory_efficient_attention()    # Lowers GPU memory usage and gives a meaningful speed up during inference
            self.prompt_compeller = self._create_prompt_compeller()         # Create compeller for prompt weighting (not available for SVD)

    def _create_prompt_compeller(self):
        """
        Used for weighing positive and negative prompts with `+`, `-`, `()` syntax: word+, word-, (two words)+, (word)#.#, etc.
        """
        try:
            import compel
            if self.model_pipe.tokenizer is not None and self.model_pipe.text_encoder is not None:
                tokenizer = [self.model_pipe.tokenizer, self.model_pipe.tokenizer_2]
                encoder = [self.model_pipe.text_encoder, self.model_pipe.text_encoder_2]
                requires = [False, True]
            else:
                # For the Img2Img pipe, trying to use `tokenizer` and `text_encoder` instead of `2` versions causes: `AttributeError: 'NoneType' object has no attribute 'device'`
                tokenizer = self.model_pipe.tokenizer_2
                encoder = self.model_pipe.text_encoder_2
                requires = True

            return compel.Compel(tokenizer=tokenizer,
                                text_encoder=encoder,
                                returned_embeddings_type=compel.ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                                requires_pooled=requires)
        except AttributeError:
            print("Could not create compeller. PROMPT WEIGHT WILL BE DISABLED.")
            return None

    def _invoke_pipe(self, prompt: str|list, negative_prompt: str|list=None, seed: int=0, images_per_prompt: int=1, inference_steps: int=40, **kwargs):
        """
        Returns generated data and seed range used to generate data.
        """
        import torch
        total_images = images_per_prompt
        if type(prompt) is list:
            total_images *= len(prompt)
        seed_range = range(seed, seed + total_images)
        seeds = [torch.Generator(device=self.model_pipe.device).manual_seed(i) for i in seed_range]

        if self.lora_paths:
            kwargs.update({"cross_attention_kwargs":{"scale":1.0}}) # Overall LoRA strength: 0 for off, 1 for max strength

        # Process using weighted prompts
        if self.prompt_compeller is not None:
            positive_embeds, positive_pooled = self.prompt_compeller(prompt)
            negative_embeds, negative_pooled = self.prompt_compeller(negative_prompt)
            return self.model_pipe(prompt_embeds=positive_embeds,
                        pooled_prompt_embeds=positive_pooled,
                        negative_prompt_embeds=negative_embeds,
                        negative_pooled_prompt_embeds=negative_pooled,
                        num_images_per_prompt=images_per_prompt,
                        num_inference_steps=inference_steps,
                        generator=seeds,
                        **kwargs).images,\
                        seed_range


    def save_outputs(self, data, seeds, source_file_name=None):
        """
        source_file_name: if provided, include this name in saved file's name
        """
        import torch

        base_name = self.output_dir
        if source_file_name is not None:    # Use the source file's base name for new files
            base_name += os.path.splitext(os.path.basename(source_file_name))[0] + "-"

        for i, seed in zip(data, seeds):
            import torch
            file_name =f"{base_name}{seed}-{str(uuid.uuid4()).split('-')[0]}"
            if type(i) is torch.Tensor:
                file_name += ".tensor"
                with open(file_name, "wb") as fp:
                    pickle.dump(i.cpu(), fp)
            elif type(i) is list:         # Save video frames
                for index, frame in enumerate(i):
                    frame.save(f"{file_name}-{index}.png")
            else:   # PIL Images
                file_name += ".png"
                i.save(file_name)
            print("Saved:", file_name)

    def generate_and_save_outputs(self, prompt, negative_prompt=None, seed=None, images_per_prompt=1, n_sets=1, inference_steps=40, source_file_name=None, **kwargs):
        """
        `kwargs` is needed for differences between arguments in pipes
        """
        if seed is None:
            seed = random.randint(0, 2**32-1)

        for i in range(n_sets):
            offset_seed = seed + i * images_per_prompt  # Makes seed sequence continuous between current set and next
            images, seed_range = self._invoke_pipe(prompt, negative_prompt, offset_seed, images_per_prompt, inference_steps, **kwargs)
            self.save_outputs(images, seed_range, source_file_name=source_file_name)
            print(f"Completed set {i + 1}/{n_sets}")


class SDXLBase(SDXLModel):
    def __init__(self, lora_dicts=[]):
        super().__init__(lora_dicts)
        self.model_dir += f"sd_xl_base_1.0{os.sep}"

        import diffusers
        self.load_model(diffusers.StableDiffusionXLPipeline, "Loading Base model:")

    def generate_and_save_latent_image(self, prompt, negative_prompt=None, seed=None, images_per_prompt=1, n_sets=1, inference_steps=40, width=768, height=768, noise=None):
        self.generate_and_save_outputs(prompt, negative_prompt, seed, images_per_prompt, n_sets, inference_steps, width=width, height=height, output_type="latent", denoising_end=noise)

class SDXLInpainting(SDXLModel):
    """
    White pixels in the mask will be repainted, while black pixels will be preserved.
    """
    def __init__(self, lora_dicts=[]):
        super().__init__(lora_dicts)
        self.model_dir += f"sd_xl_base_1.0{os.sep}"

        import diffusers
        self.load_model(diffusers.StableDiffusionXLInpaintPipeline, "Loading Inpainting model:")

    def inpaint_and_save_image(self, image, mask_image, prompt, negative_prompt=None, seed=0, images_per_prompt=1, n_sets=1, inference_steps=40):
        if type(image) is str:
            image = Image.open(image)
        if type(mask_image) is str:
            mask_image = Image.open(mask_image)

        # Removes alpha layer, if present
        image = image.convert("RGB")
        mask_image = mask_image.convert("RGB")

        self.generate_and_save_outputs(prompt, negative_prompt, seed, images_per_prompt, n_sets, inference_steps, image=image, mask_image=mask_image)


class SDXLImg2Img(SDXLModel):
    def __init__(self, model_dir=None, lora_dicts=[]):
        super().__init__(lora_dicts)
        if model_dir is None:
            self.model_dir += f"sd_xl_refiner_1.0{os.sep}"
        else:
            self.model_dir += model_dir

        import diffusers
        self.load_model(diffusers.StableDiffusionXLImg2ImgPipeline, "Loading Img2Img model:")

    def generate_and_save_from_latent_images(self, prompt, negative_prompt=None, seed=None, images_per_prompt=1, n_sets=1, inference_steps=40, noise=None, strength=0.3):
        latent_files = sorted(glob.glob(f"{self.output_dir}*.tensor"))
        print("Tensor files found:", len(latent_files))
        for i, file in enumerate(latent_files):
            print("Processing:", file)
            with open(file, "rb") as fp:
                image = pickle.load(fp)
                self.generate_and_save_outputs(prompt, negative_prompt, seed, images_per_prompt, n_sets, inference_steps, image=image, source_file_name=file, denoising_start=noise, strength=strength)
            print(f"Completed file {i + 1}/{len(latent_files)}")

    def generate_and_save_from_images(self, prompt, negative_prompt=None, seed=None, images_per_prompt=1, n_sets=1, inference_steps=40, noise=None, strength=0.3):
        pngs = sorted(glob.glob(f"{self.output_dir}*.png"))
        print("PNG files found:", len(pngs))

        # TODO process latent files in chunks of 4
        # for i in range(0, len(pngs), 4):
        #     step = pngs[i:i+4]
        #     loaded = []
        #     for p in step:
        #         print("Processing:", p)
        #         loaded.append(Image.open(p).convert("RGB"))    # Load and remove alpha layer
        #     # TODO: what to pass for `source_file_name`
        #     self.generate_and_save_outputs(prompt, negative_prompt, seed, images_per_prompt, n_sets, inference_steps, image=loaded, source_file_name=file, denoising_start=noise, strength=strength)

        for i, file in enumerate(pngs):
            print("Processing:", file)
            image = Image.open(file).convert("RGB")    # Load and remove alpha layer
            self.generate_and_save_outputs(prompt, negative_prompt, seed, images_per_prompt, n_sets, inference_steps, image=image, source_file_name=file, denoising_start=noise, strength=strength)
            print(f"Completed file {i + 1}/{len(pngs)}")


class SDXLImg2Video(SDXLModel):
    def __init__(self, model_dir=None, lora_dicts=[]):
        super().__init__(lora_dicts)
        if model_dir is None:
            self.model_dir += f"svd{os.sep}"
            # self.model_dir += f"svd_xt{os.sep}"
        else:
            self.model_dir += model_dir

        import diffusers
        self.load_model(diffusers.StableVideoDiffusionPipeline, "Loading SVD model:")

    def _invoke_pipe(self, image, seed, num_frames, num_videos, fps, inference_steps, decode_chunk_size, motion_bucket_id, noise_aug_strength):
        import torch
        seed_range = range(seed, seed + num_videos)
        seeds = [torch.Generator(device="cpu").manual_seed(i) for i in seed_range]

        return self.model_pipe(image, num_frames=num_frames, fps=fps, num_videos_per_prompt=num_videos, num_inference_steps=inference_steps, width=image.size[0], height=image.size[1], decode_chunk_size=decode_chunk_size, motion_bucket_id=motion_bucket_id, noise_aug_strength=noise_aug_strength, generator=seeds)[0], seed_range

    def generate_video_from_image(self, input_image="", width=512, height=512, seed=None, num_frames=14, num_videos=1, fps=7, inference_steps=25, decode_chunk_size=4, motion_bucket_id=127, noise_aug_strength=0.02):
        if os.path.isfile(input_image):
            pngs = [input_image]
        else:
            pngs = sorted(glob.glob(f"{self.output_dir}*.png"))

        print("PNG files found:", len(pngs))
        for i, file in enumerate(pngs):
            print("Processing:", file)
            image = Image.open(file).convert("RGB")    # Load and remove alpha layer
            image.resize((width, height))

            if seed is None:
                seed = random.randint(0, 2**32-1)

            frames, seeds = self._invoke_pipe(image, seed, num_frames, num_videos, fps, inference_steps, decode_chunk_size, motion_bucket_id, noise_aug_strength)
            self.save_outputs(frames, seeds)
            print(f"Completed video {i + 1}/{len(pngs)}")


pack_kwargs = lambda **kwargs: kwargs


def make_base_images(prompt, negative_prompt=None, seed=None, images_per_prompt=1, n_sets=1, inference_steps=40, width=768, height=768, lora_dicts=[]):
    """
    Create a set of PNGs with just the base model.
    """
    def target(**kwargs):
        sdxl = SDXLBase(lora_dicts=lora_dicts)
        sdxl.generate_and_save_outputs(**kwargs)
    keywords = pack_kwargs(prompt=prompt, negative_prompt=negative_prompt, images_per_prompt=images_per_prompt, seed=seed, n_sets=n_sets, inference_steps=inference_steps, width=width, height=height)
    proc = multiprocessing.Process(target=target, kwargs=keywords)
    proc.start()
    proc.join()


def make_latent_images(prompt, negative_prompt=None, seed=None, images_per_prompt=1, n_sets=1, inference_steps=40, width=768, height=768, noise=0.8, lora_dicts=[]):
    def target(**kwargs):
        sdxl = SDXLBase(lora_dicts=lora_dicts)
        sdxl.generate_and_save_latent_image(**kwargs)
    proc = multiprocessing.Process(target=target, kwargs=pack_kwargs(prompt=prompt, negative_prompt=negative_prompt, seed=seed, images_per_prompt=images_per_prompt, n_sets=n_sets, inference_steps=inference_steps, width=width, height=height, noise=noise))
    proc.start()
    proc.join()


def inpaint_images(image, mask_image, prompt, negative_prompt=None, seed=None, images_per_prompt=1, n_sets=1, inference_steps=40, lora_dicts=[]):
    """
    For in-painting, strength defaults to 0.999, which essentially means "completely repaint white pixel area so nothing of original is left".
    """
    def target(**kwargs):
        sdxl = SDXLInpainting(lora_dicts=lora_dicts)
        sdxl.inpaint_and_save_image(**kwargs)
    proc = multiprocessing.Process(target=target, kwargs=pack_kwargs(image=image, mask_image=mask_image, prompt=prompt, negative_prompt=negative_prompt, seed=seed, images_per_prompt=images_per_prompt, n_sets=n_sets, inference_steps=inference_steps))
    proc.start()
    proc.join()


def img_2_img(prompt, negative_prompt=None, seed=None, images_per_prompt=1, n_sets=1, inference_steps=40, strength=0.3, lora_dicts=[]):
    def target(**kwargs):
        sdxl = SDXLImg2Img(model_dir=f"sd_xl_base_1.0{os.sep}", lora_dicts=lora_dicts)   # Use base model for regular Img2Img processing
        sdxl.generate_and_save_from_images(**kwargs)
    proc = multiprocessing.Process(target=target, kwargs=pack_kwargs(prompt=prompt, negative_prompt=negative_prompt, seed=seed, images_per_prompt=images_per_prompt, n_sets=n_sets, inference_steps=inference_steps, strength=strength))
    proc.start()
    proc.join()


def make_refined_images(prompt, negative_prompt=None, inference_steps=40, noise=0.8, lora_dicts=[]):
    """
    `n_sets` > 1 and/or `images_per_prompts` > 1 produce the same identical image for a given input image and prompt
    different seeds also don't change refiner's output for a given input image and prompt; it will produce the same image with any seed
    `strength` is ignored when noise is defined
    """
    def target(**kwargs):
        sdxl = SDXLImg2Img(lora_dicts=lora_dicts)
        sdxl.generate_and_save_from_latent_images(**kwargs)
    proc = multiprocessing.Process(target=target, kwargs=pack_kwargs(prompt=prompt, negative_prompt=negative_prompt, inference_steps=inference_steps, noise=noise))
    proc.start()
    proc.join()


def make_detailed_images(prompt, negative_prompt=None, seed=None, images_per_prompt=1, n_sets=1, inference_steps=40, strength=0.3, lora_dicts={}):
    def target(**kwargs):
        sdxl = SDXLImg2Img(lora_dicts=lora_dicts)
        sdxl.generate_and_save_from_latent_images(**kwargs)
    proc = multiprocessing.Process(target=target, kwargs=pack_kwargs(prompt=prompt, negative_prompt=negative_prompt, seed=seed, images_per_prompt=images_per_prompt, n_sets=n_sets, inference_steps=inference_steps, strength=strength))
    proc.start()
    proc.join()


def img_2_video(input_image, width=1024, height=576, seed=None, num_frames=14, num_videos=1, fps=7, inference_steps=25, decode_chunk_size=4, motion_bucket_id=127, noise_aug_strength=0.02):
    """
    decode_chunk_size:  This means the VAE decodes frames in chunks instead of decoding all together; leads to slight video quality deterioration
    motion_bucket_id:   The motion bucket ID used for the generated video; this can be used to control the motion of the generated video; increasing this value will increase the motion of the generated video
    noise_aug_strength: The amount of noise added to the conditioning image; the higher the values the less the video will resemble the conditioning image; increasing this value will also increase the motion of the generated video
    """
    def target(**kwargs):
        sdxl = SDXLImg2Video()
        sdxl.generate_video_from_image(**kwargs)

    proc = multiprocessing.Process(target=target, kwargs=pack_kwargs(input_image=input_image, width=width, height=height, seed=seed, num_frames=num_frames, num_videos=num_videos, fps=fps, inference_steps=inference_steps, decode_chunk_size=decode_chunk_size, motion_bucket_id=motion_bucket_id, noise_aug_strength=noise_aug_strength))
    proc.start()
    proc.join()


def make_gif(input_folder, fps=24):
    """
    Convert set of images into a single animated GIF.
    """
    images = []
    for img in glob.glob(f"{input_folder}{os.sep}*.png"):
        images.append(Image.open(img))

    if images:
        images[0].save(f"{input_folder}{os.sep}output.gif", save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0)
    else:
        print("No images to make GIF")


if __name__ == "__main__":
    seed = None    # None = random initial seed will be used, otherwise enter an integer here; each image will get new seed of `seed += 1`
    prompt = "Astronaut riding a (white horse)1.3 on the moon" # Enter your prompt here; you can do weighted tokens as shown
    negative_prompt = "cartoon, (low quality)1.2"              # Enter your negative prompt here (things you don't want in the image); you can do weighted tokens as shown

    # Standard arguments
    width = height = 1024 # Output image dimensions; have to be divisible by 8
    inference_steps = 40  # Denoising steps; higher number makes clearer image, but takes a longer time to process
    noise = 0.85          # Used for "refined" images; base model with do `inference_steps * noise`, refiner model will do `inference_steps * (1 - noise)`
    strength = 0.2        # Used for img2img and detailed refiner, lower value means input image is changed less: actual_inference_steps = strength * inference_steps
    images_per_prompt = 4 # Images to create on the GPU in one batch; higher values are more processing time efficient, but GPU VRAM usage increases; lower this if out of memory errors occur
    n_sets = 2            # Rerun the same prompt this many times: total images produced = `images_per_prompt` * `n_sets`
                          # When `images_per_prompt` can't be made any higher (no GPU memory left), increase this value to get the total images you want for the prompt

    # DETAILED refiner arguments
    detailed_images_per_prompts = 1 # Detailed image variants to create on the GPU at one time
                                    # Refiner uses more VRAM per image, so it can't be as high as the base model's `images_per_prompt`
    refiner_n_sets = 1     # Increases the total number of detailed variants by running the same prompt and input image this many times
    refiner_seed = seed    # The refiner's seed is uncorrelated to the base model's, so you can make it what you want;
                           # making it the same as the base image seed doesn't have any affect besides being convenient for record keeping

    # Specify LoRAs to load from `models/lora`; leave list empty if you don't want to use any
    # Will only be used with the base SDXL model; the refiner will not use them
    # `scale` is how strong the LoRA will be applied: 0 to 1.0
    base_lora_dicts = [
        {
        "model":"lora_file_name_without_file_extension",
        "scale":0.8
        },
        {
        "model":"another_lora",
        "scale":0.5
        },
    ]

    # MAKE PNGS USING JUST THE BASE MODEL
    make_base_images(prompt, negative_prompt, seed, images_per_prompt, n_sets, inference_steps, width, height, base_lora_dicts)

    # IN-PAINT IMAGES
    input_image = "outputs/image.png"
    mask_image = "outputs/mask.png"     # Replace white areas, keep black areas unchanged; decrease `inference_steps` to decrease strength of replacement
    # inpaint_images(input_image, mask_image, prompt, negative_prompt, seed, images_per_prompt, n_sets, inference_steps, base_lora_dicts)

    # IMG 2 IMG
    # Loads images stored in the output folder and converts them to new images using the prompt
    # NOTE `images_per_prompt` > 1 seems to just generate the same image each time, so argument not used here
    # img_2_img(prompt, negative_prompt, seed, 1, n_sets, inference_steps, strength, base_lora_dicts)

    # MAKE REFINED PNGS USING THE REFINER MODEL
    # 1) Create latent images from base model (with added noise)
    # 2) Feed noisy latent images into the refiner to make a more refined PNG; images are theoretically closer to the original prompt compared to base mode alone
    # make_latent_images(prompt, negative_prompt, seed, images_per_prompt, n_sets, inference_steps, width, height, noise, base_lora_dicts)
    # make_refined_images(prompt, negative_prompt, inference_steps, noise)

    # MAKE DETAILED PNGS USING THE REFINER MODEL
    # 1) Create latent images from base model (no noise added)
    # 2) Feed latent images into the refiner to make PNGs with finer details
    # You can create multiple output variants of the input image by increasing `detailed_images_per_prompts` and `refiner_n_sets`
    # These images will look similar to each, but have different minor details
    # make_latent_images(prompt, negative_prompt, seed, images_per_prompt, n_sets, inference_steps, width, height, None, base_lora_dicts)
    # make_detailed_images(prompt, negative_prompt, refiner_seed, detailed_images_per_prompts, refiner_n_sets, inference_steps, strength)

    # ANIMATE IMAGE WITH SVD
    # FIXME not currently working (low VRAM?)
    num_videos = 1
    width, height = 512, 512
    num_frames = 2
    fps = num_frames
    inference_steps = 25        # 25 is still very blurry
    decode_chunk_size = 1       # Lower = lower VRAM, more distortion
    motion_bucket_id = 127      # Higher = more motion; Default = 127
    noise_aug_strength = 0.02   # Higher = more different video + more motion; Default = 0.02
    # img_2_video("image.png", width, height, seed, num_frames, fps, num_videos, inference_steps, decode_chunk_size, motion_bucket_id, noise_aug_strength)
    # make_gif("outputs/fixed_face/", fps)
